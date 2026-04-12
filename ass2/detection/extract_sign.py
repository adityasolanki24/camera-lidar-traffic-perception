# extract_sign.py
# Given detected traffic cones, crop each cone and try to extract the sign region.
# Main idea:
# 1) crop the cone
# 2) look for black/white square-like regions inside it
# 3) keep the best candidate as the sign
# 4) optionally save the sign crop for debugging

# Imports
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

# --- extract_blue_sign_region ----------------------------------------------
# Create a blue mask to isolate the arrow symbol region inside a cone crop
def extract_blue_sign_region(cylinder_crop: np.ndarray) -> np.ndarray:
    if cylinder_crop.size == 0 or cylinder_crop.shape[0] < 2 or cylinder_crop.shape[1] < 2:
        return np.zeros((1, 1), dtype=np.uint8)

    hsv = cv2.cvtColor(cylinder_crop, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 70, 40], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    return blue_mask


# --- extract_sign_region ---------------------------------------------------
# Create a black/white mask to isolate the likely sign region inside a cone crop
def extract_sign_region(cylinder_crop: np.ndarray) -> np.ndarray:
    # Return a tiny blank mask if the crop is invalid
    if cylinder_crop.size == 0 or cylinder_crop.shape[0] < 2 or cylinder_crop.shape[1] < 2:
        return np.zeros((1, 1), dtype=np.uint8)

    # Convert cone crop from BGR to HSV for colour thresholding
    hsv = cv2.cvtColor(cylinder_crop, cv2.COLOR_BGR2HSV)

    # Threshold range for white sign areas
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)

    # Threshold range for black sign areas
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 70], dtype=np.uint8)

    # Mask white pixels
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Mask black pixels
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Combine black and white masks into one binary mask
    return cv2.bitwise_or(white_mask, black_mask)


# --- find_blue_sign_candidates ---------------------------------------------
# Find blue arrow/circle regions that likely belong to a traffic sign
def find_blue_sign_candidates(
    mask: np.ndarray, min_area: int = 80
) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sign_candidates: list[tuple[int, int, int, int]] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue

        aspect_ratio = w / float(h)
        if 0.45 < aspect_ratio < 1.8 and w * h >= min_area:
            sign_candidates.append((x, y, w, h))

    return sign_candidates


# --- find_square_sign_candidates -------------------------------------------
# Find square-like bounding boxes in a binary mask
def find_square_sign_candidates(
    mask: np.ndarray, min_area: int = 500
) -> list[tuple[int, int, int, int]]:
    # Find connected contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store valid sign candidates as bounding boxes
    sign_candidates: list[tuple[int, int, int, int]] = []

    # Check each contour
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Reject empty boxes
        if w <= 0 or h <= 0:
            continue

        # Compute width-to-height ratio
        aspect_ratio = w / float(h)

        # Keep only roughly square regions that are large enough
        if 0.7 < aspect_ratio < 1.3 and w * h > min_area:
            sign_candidates.append((x, y, w, h))

    return sign_candidates


# --- find_square_sign_candidates_canny -------------------------------------
# Fallback method: use Canny edges instead of HSV black/white masking
def find_square_sign_candidates_canny(
    cylinder_crop: np.ndarray, min_area: int = 500
) -> list[tuple[int, int, int, int]]:
    # Return no candidates if crop is invalid
    if cylinder_crop.size == 0:
        return []

    # Convert crop to grayscale for edge detection
    gray = cv2.cvtColor(cylinder_crop, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 100, 200)

    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store valid sign candidates as bounding boxes
    sign_candidates: list[tuple[int, int, int, int]] = []

    # Check each contour
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Reject empty boxes
        if w <= 0 or h <= 0:
            continue

        # Compute width-to-height ratio
        aspect_ratio = w / float(h)

        # Keep only roughly square regions that are large enough
        if 0.7 < aspect_ratio < 1.3 and w * h > min_area:
            sign_candidates.append((x, y, w, h))

    return sign_candidates


# --- pick_largest_bbox -----------------------------------------------------
# From all sign candidates, keep the largest one
def pick_largest_bbox(
    candidates: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    # Return nothing if no candidates were found
    if not candidates:
        return None

    # Return the candidate with the largest box area
    return max(candidates, key=lambda b: b[2] * b[3])


# --- extract_sign_from_cylinder --------------------------------------------
# Extract the most likely sign crop from one cone crop
def extract_sign_from_cylinder(
    cylinder_crop: np.ndarray,
    min_area: int = 250,
) -> tuple[np.ndarray | None, str, np.ndarray | None, tuple[int, int, int, int] | None]:
    """
    Returns:
    (sign_crop_bgr, method_label, debug_mask_or_none, bbox_in_crop)

    method_label:
    - 'blue'  -> sign found using blue sign-region detection
    - 'hsv'   -> sign found using black/white HSV masking
    - 'canny' -> sign found using Canny fallback
    - 'none'  -> no valid sign found
    """

    # Return no result if the cone crop is invalid
    if cylinder_crop.size == 0:
        return None, "none", None, None

    # Get crop size
    h, w = cylinder_crop.shape[:2]

    # Adaptive thresholds help keep small distant signs while rejecting tiny noise.
    min_area = max(min_area, int(0.008 * h * w))
    min_final_area = max(350, int(0.015 * h * w))

    # First, look for the blue symbol region then expand to the square sign.
    blue_roi_y0 = int(0.08 * h)
    blue_roi_y1 = int(0.86 * h)
    blue_roi = cylinder_crop[blue_roi_y0:blue_roi_y1, :]
    blue_mask = extract_blue_sign_region(blue_roi)
    blue_candidates = find_blue_sign_candidates(blue_mask, min_area=max(80, min_area))

    bw_mask = None
    candidates: list[tuple[int, int, int, int]] = []
    method = "none"
    use_roi_offset = False
    roi_y0 = 0

    if blue_candidates:
        bx, by, bw, bh = pick_largest_bbox(blue_candidates)
        center_x = bx + bw / 2.0
        center_y = by + bh / 2.0 + blue_roi_y0
        side = int(round(max(bw, bh) * 1.55))
        sx = int(round(center_x - side / 2.0))
        sy = int(round(center_y - side / 2.0))
        candidates = [(sx, sy, side, side)]
        method = "blue"
    else:
        #  HSV search as the main fallback for clean square detections
        roi_y0 = int(0.3 * h)
        roi_y1 = int(0.8 * h)
        roi = cylinder_crop[roi_y0:roi_y1, :]
        bw_mask = extract_sign_region(roi)
        candidates = find_square_sign_candidates(bw_mask, min_area=min_area)
        method = "hsv"
        use_roi_offset = True

        # If HSV fails try Canny edge detection on the full cone crop
        if not candidates:
            candidates = find_square_sign_candidates_canny(cylinder_crop, min_area=min_area)
            method = "canny"
            bw_mask = None
            use_roi_offset = False

    # Choose the largest sign candidate
    best = pick_largest_bbox(candidates)

    # Return no sign if nothing valid was found
    if best is None:
        return None, "none", bw_mask, None

    # Unpack candidate box
    sx, sy, sw, sh = best

    # Shift y back up only when the candidate came from the HSV ROI
    if use_roi_offset:
        sy = sy + roi_y0

    # Blue detections are already expanded to the surrounding square sign
    if method != "blue":
        pad_x = int(0.12 * sw)
        pad_y = int(0.12 * sh)

        sx = sx - pad_x
        sy = sy - pad_y
        sw = sw + 2 * pad_x
        sh = sh + 2 * pad_y

    # Clamp sign box to stay inside the cone crop
    ch, cw = cylinder_crop.shape[:2]
    sx = max(0, min(sx, cw - 1))
    sy = max(0, min(sy, ch - 1))
    sw = min(sw, cw - sx)
    sh = min(sh, ch - sy)

    # Reject invalid boxes
    if sw <= 0 or sh <= 0:
        return None, "none", bw_mask, None

    # Reject very small noisy detections
    if sw * sh < min_final_area:
        return None, "none", bw_mask, None

    # Crop the final sign region
    sign_crop = cylinder_crop[sy: sy + sh, sx: sx + sw]
    sign_crop = tighten_sign_crop(sign_crop, method)

    return sign_crop, method, bw_mask, (sx, sy, sw, sh)


# --- tighten_sign_crop -----------------------------------------------------
# Trim oversized blue-based crops so the classifier sees mostly the sign.
def tighten_sign_crop(sign_crop: np.ndarray | None, method: str) -> np.ndarray | None:
    if sign_crop is None or sign_crop.size == 0 or method != "blue":
        return sign_crop

    # Keep tiny distant signs as-is; only tighten obviously oversized blue crops.
    if max(sign_crop.shape[:2]) < 90:
        return sign_crop

    hsv = cv2.cvtColor(sign_crop, cv2.COLOR_BGR2HSV)
    strict_blue_mask = cv2.inRange(
        hsv,
        np.array([100, 120, 40], dtype=np.uint8),
        np.array([132, 255, 255], dtype=np.uint8),
    )
    strict_blue_mask = cv2.morphologyEx(
        strict_blue_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    strict_blue_mask = cv2.morphologyEx(
        strict_blue_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )

    contours, _ = cv2.findContours(
        strict_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return sign_crop

    best_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best_contour)
    if w <= 0 or h <= 0:
        return sign_crop

    # If the strict-blue refinement collapses to a tiny fragment, keep the
    # original blue crop instead of over-trimming it.
    if (w * h) < 0.02 * (sign_crop.shape[0] * sign_crop.shape[1]):
        return sign_crop

    side = int(round(max(w, h) * (1.8 if max(w, h) < 20 else 1.45)))
    center_x = x + w / 2.0
    center_y = y + h / 2.0

    crop_x0 = max(0, int(round(center_x - side / 2.0)))
    crop_y0 = max(0, int(round(center_y - side / 2.0)))
    crop_x1 = min(sign_crop.shape[1], crop_x0 + side)
    crop_y1 = min(sign_crop.shape[0], crop_y0 + side)

    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return sign_crop

    tightened_crop = sign_crop[crop_y0:crop_y1, crop_x0:crop_x1]
    if tightened_crop.size == 0:
        return sign_crop

    return tightened_crop


# --- safe_crop -------------------------------------------------------------
# Crop an image safely without going outside image bounds
def safe_crop(image_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    
    # Return empty if the input image is invalid
    if image_bgr is None or image_bgr.size == 0:
        return np.array([])
    
    # Get image size
    img_h, img_w = image_bgr.shape[:2]

    # Clamp crop start coordinates to stay inside the image
    x0 = max(0, x)
    y0 = max(0, y)

    # Clamp crop end coordinates to stay inside the image
    x1 = min(img_w, x + w)
    y1 = min(img_h, y + h)
    
    # Return empty if the crop is invalid
    if x1 <= x0 or y1 <= y0:
        return np.array([])
    
    # Return a copy of the cropped region
    return image_bgr[y0:y1, x0:x1].copy()


# --- save_sign_crop --------------------------------------------------------
# Save one extracted sign crop if valid
def save_sign_crop(
    sign_crop: np.ndarray | None,
    output_dir: str | Path,
    image_stem: str,
    cone_id: int,
    method: str,
) -> str | None:
    # Skip saving if the crop is invalid
    if sign_crop is None or sign_crop.size == 0:
        return None

    # Ensure output folder exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output filename
    output_name = f"{image_stem}_cone{cone_id}_{method}.png"
    output_path = output_dir / output_name

    # Save the crop
    cv2.imwrite(str(output_path), sign_crop)

    return str(output_path)


# --- extract_signs_from_detections ----------------------------------------
# Extract signs from already-detected cones in one image
def extract_signs_from_detections(
    image_bgr: np.ndarray,
    cone_detections: list[dict],
    save_crops: bool = False,
    output_dir: str | Path | None = None,
    image_stem: str | None = None,
) -> list[dict]:
    # Store one result per detected cone
    sign_results = []

    # Process each detected cone
    for cone_detection in cone_detections:
        contour = cone_detection["contour"]
        cone_id = cone_detection["cone_id"]

        # Get the cone bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the cone from the image
        cylinder_crop = safe_crop(image_bgr, x, y, w, h)

        # Extract the sign from the cone crop
        sign_crop, method, debug_mask, bbox_in_crop = extract_sign_from_cylinder(cylinder_crop)

        # Optionally save the sign crop for debugging
        saved_path = None
        if save_crops and output_dir is not None and image_stem is not None:
            saved_path = save_sign_crop(
                sign_crop,
                output_dir,
                image_stem,
                cone_id,
                method,
            )

        # Store the final sign extraction result for this cone
        sign_results.append(
            {
                "cone_id": cone_id,
                "cone_bbox": (x, y, w, h),
                "sign_crop": sign_crop,
                "sign_method": method,
                "sign_debug_mask": debug_mask,
                "sign_bbox_in_crop": bbox_in_crop,
                "saved_sign_crop_path": saved_path,
            }
        )

    return sign_results
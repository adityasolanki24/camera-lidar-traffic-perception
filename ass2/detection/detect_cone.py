# detect_cylinder.py 
# It detects traffic cones in the image by the following process: 
# 1) Uses HSV mask to find all the red objects - Note: It is possible that the HSV mask finds red objects that are not cones. 
# Therefore, to improve the robustness, the process does not rely mainly on HSV colour masking. 
# 2) Obtains the contour around the detected red objects and draws a bounding box rectangle 
# 3) Checks whether the the bounding box height > width. If false, reject this contour. 
# 4) Estimates a line along the left and right sides of the contour. Check if this line is near-vertical. If false, reject this contour
#--------------------------------------

# Imports
from __future__ import annotations
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Output paths 
GRID_OUTPUT_PATH = Path("visualise_results.png")


def detect_cone(dataset_path: str | Path):
    # Load in the image dataset 
    dataset_path = Path(dataset_path)
    # Checking if the dataset path was given correctly, otherwise raise error 
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
   
    # Create an array that stores the images 
    img_dataset = []
    image_filenames = []
    for file_path in sorted(dataset_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() == ".png":
            image_bgr = cv2.imread(str(file_path))
            img_dataset.append(image_bgr)
            image_filenames.append(file_path.name)

    #----------------------- MAIN IMAGE PROCESSING LOGIC FOR CYLINDER DETECTION ---------------------------------------------# 

    # For every image in the image dataset, 1) convert from BGR to HSV, and 2) apply the HSV segmentation mask, 
    # 3) draw a bounding box around the pixels that are TRUE from the mask

    # At every step of the method, we will save the output to processed_dataset so we can visualise later. 
    processed_dataset = []
    cone_detections_by_image = {}
    for image_filename, image_bgr in zip(image_filenames, img_dataset):
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)   # we want to view images in HSV format
        
        # Create the binary mask for red pixels by only checking pixels that fall in the HSV ranges as selected (see below)
        red_mask = create_red_cylinder_mask(image_hsv)     
        
        # Find the contour around the red pixels 
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Since it possible that we may have found red objects that are not the traffic cylinders, 
        # we further check two conditions on this contour: 1) its bounding box height > width, 
        # 2) compact within its bounding box, 3) 
        cylinder_contours = [] # will only have contours that pass all the conditions 
        for contour in contours:
            box = cv2.boundingRect(contour) # compute the reactangle bounding box 
        # the contour should pass these conditions to classify as the traffic cylinder
            if contour_looks_tall_and_compact(contour, box) and contour_has_vertical_side_structure(contour, box):
                cylinder_contours.append(contour)
        
        #--------------------- THIS IS USED FOR DISTANCE POSITION -----------------------------------------------#
        cone_detections = build_cone_detections(cylinder_contours)
        #--------------------------------------------------------------------------------------------------------#
        processed_dataset.append(
            {
                "image_filename": image_filename,
                "image_bgr": image_bgr,
                "image_hsv": image_hsv,
                "red_mask": red_mask,
                "contours": contours,
                "cone_detections": cone_detections,
            }
        )
        cone_detections_by_image[image_filename] = cone_detections

    # Save the outputs onto grid image so we can visualise easily 
    visualise_results(processed_dataset, GRID_OUTPUT_PATH)
    print(f"Saved 6 x 6 extraction grid to: {GRID_OUTPUT_PATH}")
    return cone_detections_by_image


# Creates the binary mask for traffic cylinder pixels 
# The mask only has the pixels that are inside the chosen red HSV ranges
# the red HSV ranges were chosen by VISUALLY INSPECTING the generated HSV plot from hsv_plot.py 
def create_red_cylinder_mask(image_hsv: np.ndarray):
    # The colour red wraps around the HSV colour space, therefore we have a lower and upper range 
    lower_red_1 = np.array([0, 160, 60], dtype=np.uint8)
    upper_red_1 = np.array([15, 255, 255], dtype=np.uint8)

    lower_red_2 = np.array([155, 160, 60], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

    mask_1 = cv2.inRange(image_hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(image_hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)   # Combine the lower and higher ranges 

    # Morphological steps to clean up the mask 
    open_kernel = np.ones((5, 5), np.uint8) 
    close_kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel) # Removes small noisy blobs. it removes tiny false red detections
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel) # Fills small black gaps inside the cylinder
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


# Check if bounding box height > width and the contour is compact within its bounding box 
def contour_looks_tall_and_compact(contour: np.ndarray, box: tuple[int, int, int, int]):
    x, y, width, height = box
    if height <= width:
        return False

    bounding_box_area = width * height
    if bounding_box_area == 0:
        return False

    contour_area = cv2.contourArea(contour)
    compactness_ratio = contour_area / bounding_box_area

    return compactness_ratio >= 0.45

# Checks whether a contour looks like it has left and right vertical sides 
def get_contour_side_fit_lines(
    contour: np.ndarray, box: tuple[int, int, int, int]
):
    x, y, width, height = box
    # Reject the contour if it does not pass height > width 
    if height <= width:
        return (False, None, None)

    # Reshape the contour boundary into (x,y) points 
    contour_points = contour.reshape(-1, 2).astype(np.float32)
    if len(contour_points) < 6:
        return (False, None, None)

    # Simplifies the contour so the line fit is less noisy 
    epsilon = 0.01 * cv2.arcLength(contour, True)
    simplified_contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float32)
    if len(simplified_contour) >= 6:
        contour_points = simplified_contour

    # Split the contour points into left-side and right-side points based on whether a point is left or right of the contour's average x 
    center_x = float(np.mean(contour_points[:, 0]))
    left_points = contour_points[contour_points[:, 0] <= center_x]
    right_points = contour_points[contour_points[:, 0] >= center_x]

    if len(left_points) < 2 or len(right_points) < 2:
        return (False, None, None)

    # Fit a line to the left points and to the right points using cv2.fitLine 
    def fit_vertical_side(points: np.ndarray):
        x_values = points[:, 0]
        y_values = points[:, 1]
        # Find the distance between the topmost point to the bottommost point of that side in the vertical (y) direction
        vertical_span = float(np.max(y_values) - np.min(y_values))
        # If this vertical span only exists for a tiny section of the contour, then it is not reliable cylinder side
        if vertical_span < 0.3 * height:
            return (False, False, 0.0, 0.0, None) # Exit 

        
        # Fit the line along the points of the side 
        line = cv2.fitLine(points.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(line[0, 0])
        vy = float(line[1, 0])
        line_x = float(line[2, 0])
        line_y = float(line[3, 0])

        # If vy is zero then the line is horizontal and not a vertical-side 
        if abs(vy) < 1e-6:
            return (False, False, 0.0, 0.0, None)

        slope = vx / vy
        predicted_x = line_x + slope * (y_values - line_y) # This is the line that was estimated by cv2.fitLine
        
        # We want to measure how far the real contour points are from that fitted line, in the horizontal direction. 
        mean_x_error = float(np.mean(np.abs(x_values - predicted_x)))
        
        # Decide if it's a strong fit or not based on the mean error 
        strong_fit = mean_x_error <= max(5.5, 0.18 * width)
        weak_fit = mean_x_error <= max(8.0, 0.28 * width)
        if not weak_fit:
            return (False, False, 0.0, 0.0, None)

        top_y = float(np.min(y_values))
        bottom_y = float(np.max(y_values))
        top_x = float(line_x + slope * (top_y - line_y))
        bottom_x = float(line_x + slope * (bottom_y - line_y))
        line_points = (
            (int(round(top_x)), int(round(top_y))),
            (int(round(bottom_x)), int(round(bottom_y))),
        )

        return (True, strong_fit, slope, vertical_span, line_points)
    
    # Run the function on the left and right sides of the contour 
    left_ok, left_strong, left_slope, left_span, left_line = fit_vertical_side(left_points)
    right_ok, right_strong, right_slope, right_span, right_line = fit_vertical_side(right_points)

    if not left_ok and not right_ok:
        return (False, left_line, right_line)

    if left_ok and right_ok:
        slopes_are_parallel = abs(left_slope - right_slope) <= 0.45
        at_least_one_strong = left_strong or right_strong
        return (slopes_are_parallel and at_least_one_strong, left_line, right_line)

    if left_ok and left_strong and left_span >= 0.55 * height:
        return (True, left_line, right_line)

    if right_ok and right_strong and right_span >= 0.55 * height:
        return (True, left_line, right_line)

    return (False, left_line, right_line)


# Calls the get_contour_side_fit_lines function and keeps only whether the contour passes the side-structure test
def contour_has_vertical_side_structure(contour: np.ndarray, box: tuple[int, int, int, int]):
    passes_structure, _, _ = get_contour_side_fit_lines(contour, box)
    return passes_structure


def find_center_point(contour: np.ndarray):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        x, y, width, height = cv2.boundingRect(contour)
        return (x + width // 2, y + height // 2)

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return (center_x, center_y)


def build_cone_detections(cylinder_contours: list[np.ndarray]):
    cone_detections = []

    for contour in cylinder_contours:
        center_point = find_center_point(contour)
        cone_detections.append(
            {
                "contour": contour,
                "center_point": center_point,
            }
        )

    cone_detections.sort(key=lambda cone_detection: cone_detection["center_point"][0])

    for cone_id, cone_detection in enumerate(cone_detections, start=1):
        cone_detection["cone_id"] = cone_id

    return cone_detections


#---------------------- FOR VISUALISATION ----------------------------------------------_#


def draw_red_blob_bounding_boxes(image_bgr: np.ndarray, contours: list[np.ndarray]):
    boxed_image = image_bgr.copy()

    for contour in contours:
        cv2.drawContours(boxed_image, [contour], -1, (0, 255, 255), 2)
    return boxed_image


def draw_fitted_line_visualization(image_bgr: np.ndarray, contours: list[np.ndarray]):
    fitted_line_image = image_bgr.copy()

    for contour in contours:
        box = cv2.boundingRect(contour)
        _, left_line, right_line = get_contour_side_fit_lines(contour, box)

        cv2.drawContours(fitted_line_image, [contour], -1, (0, 255, 255), 2)

        if left_line is not None:
            cv2.line(fitted_line_image, left_line[0], left_line[1], (255, 255, 0), 3)
        if right_line is not None:
            cv2.line(fitted_line_image, right_line[0], right_line[1], (255, 255, 0), 3)

    return fitted_line_image


def draw_cylinder_bounding_boxes(
    image_bgr: np.ndarray,
    cone_detections: list[dict[str, object]],
):
    boxed_image = image_bgr.copy()

    for cone_detection in cone_detections:
        contour = cone_detection["contour"]
        x, y, width, height = cv2.boundingRect(contour)
        center_point = cone_detection["center_point"]
        cone_id = cone_detection["cone_id"]
        cv2.rectangle(boxed_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.circle(boxed_image, center_point, 6, (0, 255, 255), -1)
        cv2.putText(
            boxed_image,
            str(cone_id),
            (center_point[0] + 8, center_point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return boxed_image


def convert_bgr_to_rgb(image_bgr: np.ndarray):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Outputs are neatly placed in a grid to show the original image and the processed image outputs 
def visualise_results(processed_dataset: list[dict[str, np.ndarray]], output_path: Path):
    num_image_sets = len(processed_dataset)
    num_columns = 4
    figure, axes = plt.subplots(num_image_sets, num_columns, figsize=(16, 3.5 * num_image_sets))
    axes = np.atleast_2d(axes)
    axes_flat = axes.flatten()

    for axis in axes_flat:
        axis.axis("off")

    for index, result in enumerate(processed_dataset, start=1):
        # Apply AND operation on the original image (image_bgr) and the red pixels (red_mask) to segment out only red objects
        cylinder_segmented = cv2.bitwise_and(result["image_bgr"], result["image_bgr"], mask=result["red_mask"])
        raw_boxed_image = draw_red_blob_bounding_boxes(result["image_bgr"], result["contours"])
        fitted_line_visualization = draw_fitted_line_visualization(result["image_bgr"], result["contours"])
        boxed_image = draw_cylinder_bounding_boxes(
            result["image_bgr"],
            result["cone_detections"],
        )

        panels = [
            ("HSV Thresholding for Red Objects", convert_bgr_to_rgb(cylinder_segmented)),
            ("Contouring Detected Red Objects", convert_bgr_to_rgb(raw_boxed_image)),
            ("Estimated Left and Right Edges from Contour", convert_bgr_to_rgb(fitted_line_visualization)),
            ("Detected Cones", convert_bgr_to_rgb(boxed_image)),
        ]

        row_axes = axes[index - 1]
        row_axes[0].text(
            0.5,
            1.18,
            f"Image {index}",
            transform=row_axes[0].transAxes,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

        for panel_offset, (title, panel_image) in enumerate(panels):
            axis = row_axes[panel_offset]
            axis.imshow(panel_image)
            axis.set_title(title, fontsize=8)
            axis.axis("off")

    plt.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

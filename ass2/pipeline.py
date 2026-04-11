# ========================= PIPELINE ==========================================================

# Library imports 
from pathlib import Path
import cv2

# File imports
from classification.model import TrafficSignClassifier
from detection.detect_cone import detect_cone
from detection.detect_distance import LidarInterface
from detection.extract_sign import extract_signs_from_detections

# Input data folders
IMAGE_FOLDER = Path("traffic_cone_images")
LIDAR_FOLDER = Path("traffic_cone_lidar")
CALIBRATION_JSON = Path("calibration.json")

# Output data folders
LIDAR_OUTPUT_FOLDER = Path("lidar_detection_outputs")
SIGN_OUTPUT_FOLDER = Path("extracted_sign_outputs")

# Variables
LIDAR_FILTER_THRESHOLD = 0.15

# ========================== OUTPUT HELPERS ================================================
def clear_sign_output_folder(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()

# ============================= MAIN ==========================================================
def main():
    
    # Extract data from any ros bag
    
    
    # Load Lidar interface
    lidar_interface = LidarInterface(LIDAR_FOLDER, CALIBRATION_JSON)

    # Detect cones in all images
    cone_detections_by_image = detect_cone(IMAGE_FOLDER)

    classifier = TrafficSignClassifier()

    # Create folder for output images
    LIDAR_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    clear_sign_output_folder(SIGN_OUTPUT_FOLDER)

    # Loop through the images and test the matching LiDAR loading 
    for image_filename, cone_detections in cone_detections_by_image.items():
        try:
            image_bgr, lidar_points = load_image_and_lidar(
                image_filename,
                lidar_interface
            )
            
        # Data not found 
        except Exception as error:
            print(f"Could not load data for {image_filename}: {error}")
            continue

        # Project LiDAR points into image space
        projected_points, valid_camera_points = lidar_interface.project_points_to_image(
            lidar_points
        )

        # Get LiDAR distance statistics for this image
        cone_stats = lidar_interface.get_cone_distance_stats(
            cone_detections,
            projected_points,
            valid_camera_points,
            LIDAR_FILTER_THRESHOLD
        )

        # Extract the signs from already-detected cones
        sign_results = extract_signs_from_detections(
            image_bgr,
            cone_detections,
            save_crops=True,
            output_dir=SIGN_OUTPUT_FOLDER,
            image_stem=Path(image_filename).stem,
        )
        
        # Merge the results 
        cone_data = merge_cone_results(cone_stats, sign_results)

        classifier.classify_cone_data(cone_data)
        
        # Print the results for this image
        print_cone_stats(
            image_filename,
            cone_detections,
            cone_stats
        )
        print_sign_classifications(cone_data)

        # Save debug image showing which projected points fall inside each contour
        lidar_interface.save_lidar_distance_image(
            LIDAR_OUTPUT_FOLDER / f"{Path(image_filename).stem}_contour_lidar.png",
            image_bgr,
            cone_stats
        )

        # Output final result
        
        
# ========================== HELPER FUNCTIONS ============================================= 
# --- load_image_and_lidar -------------------------------------------------
# Load one image and its matching LiDAR cloud
def load_image_and_lidar(
    image_filename: str,
    lidar_interface: LidarInterface,
):
    image_path = IMAGE_FOLDER / image_filename
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"Could not load image: {image_filename}")

    lidar_points = lidar_interface.load_matching_point_cloud(image_filename)

    return image_bgr, lidar_points

# --- print_cone_stats -----------------------------------------------------
# Print cone distance statistics for one image
def print_cone_stats(
    image_filename: str,
    cone_detections: list[dict],
    cone_stats: list[dict],
):
    print(f"\nImage: {image_filename}")
    print(f"Cones detected: {len(cone_detections)}")

    for cone_stat in cone_stats:
        cone_id = cone_stat["cone_id"]
        mean_distance = cone_stat["mean_distance"]
        std_distance = cone_stat["std_distance"]
        
        if mean_distance is None:
            print(
                f"  Cone {cone_id}: "
                f"no LiDAR after filtering"
            )
            
        else:
            print(
                f"  Cone {cone_id}: "
                f"mean distance = {mean_distance:.3f} m, "
                f"std distance = {std_distance:.3f} m"
            )
            
# --- merge_cone_results ---------------------------------------------------
# Merge cone and LiDAR results with sign extraction using cone_id
def merge_cone_results(
    cone_stats: list[dict],
    sign_results: list[dict],
) -> list[dict]:
    
    # Index sign results by cone_id for fast lookup
    sign_results_by_id = {
        sign_result["cone_id"]: sign_result
        for sign_result in sign_results
    }

    merged_results = []

    for cone_stat in cone_stats:
        cone_id = cone_stat["cone_id"]
        sign_result = sign_results_by_id.get(cone_id, {})

        merged_result = {
            **cone_stat,
            "sign_crop": sign_result.get("sign_crop"),   # Sign in BGR format 
            "saved_sign_crop_path": sign_result.get("saved_sign_crop_path"), # Path to cropped image file
        }

        merged_results.append(merged_result)

    return merged_results


def print_sign_classifications(cone_data: list[dict]) -> None:
    for row in cone_data:
        cone_id = row.get("cone_id")
        name = row.get("predicted_class_name")
        conf = row.get("classification_confidence")
        if name is None:
            print(f"  Cone {cone_id}: sign — no crop / not classified")
        else:
            print(
                f"  Cone {cone_id}: sign — {name} "
                f"(confidence {conf:.3f})"
            )


if __name__ == "__main__":
    main()

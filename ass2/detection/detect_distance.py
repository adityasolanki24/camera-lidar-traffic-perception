# DETECT_DISTANCE: Detect cone distance using LiDAR data 
# Contains functions to data match image and lidar data of the same file name. Remove 
# null data points. Convert from robot optical frame to opencv optical frame. Then
# project the lidar points onto the camera frame using the calibration camera information.
# Then it filters data and sorts to find the closes cluster of lidar data to get the mean 
# and standard deviation of the lidar points that allign with a detected cone contour. The
# lidar points within the contour, and metrics are shown on the original image and stored 
# in a given output folder path input in main. 

# Library imports 
from __future__ import annotations
from pathlib import Path
import numpy as np
import json
import open3d as o3d
import cv2

# ==== LIDAR INTERFACE ======================================================
class LidarInterface:
    # --- ___init___ -------------------------------------------------
    # Initialise and link to lidar data and the calibration data 
    def __init__(self, lidar_folder: str | Path, calibration_json: str | Path):
        self.lidar_folder = Path(lidar_folder)
        self.calibration_json = Path(calibration_json)

        # Initialise the calibration characteristics 
        self.camera_matrix = None
        self.dist_coeffs = None
        self.lidar_to_camera_2d = None

        self.load_calibration()

    # --- load_calibration -------------------------------------------------
    # Fill out calibration characteristics with json data 
    def load_calibration(self) -> None:
        # If no json avaliable send error message
        if not self.calibration_json.exists():
            raise FileNotFoundError(f"Calibration file does not exist: {self.calibration_json}")

        # Read json file
        with open(self.calibration_json, "r") as file:
            calibration_data = json.load(file)

        # Extract and assign data 
        self.camera_matrix = np.array(calibration_data["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(calibration_data["dist_coeffs"], dtype=np.float64)
        self.lidar_to_camera_2d = np.array(calibration_data["lidar_to_camera_2d"], dtype=np.float64)

    # --- load_point_cloud -------------------------------------------------
    # Get the point cloud data from a given path to the file
    def load_point_cloud(self, point_cloud_path: str | Path) -> np.ndarray:
        # Input path to point cloud data 
        point_cloud_path = Path(point_cloud_path)

        # Check to see if the file path is valid 
        if not point_cloud_path.exists():
            raise FileNotFoundError(f"Point cloud file does not exist: {point_cloud_path}")

        # Get the N x 3 array of lidar points 
        point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))
        points = np.asarray(point_cloud.points, dtype=np.float64)

        # Remove zero rows 
        non_zero_mask = ~np.all(np.isclose(points, 0.0), axis=1)
        points = points[non_zero_mask]

        # Raise error if now valid points remain
        if len(points) == 0:
            raise ValueError(f"Point cloud is empty: {point_cloud_path}")

        return points

    # --- get_lidar_path_from_image_name --------------------------------------------
    # From a given image filename and access the corresponding lidar file
    def get_lidar_path_from_image_name(self, image_filename: str) -> Path:
        # Get corresponding lidar for image 
        image_stem = Path(image_filename).stem
        point_cloud_path = self.lidar_folder / f"{image_stem}.pcd"

        # If no matching filename exists throw an error 
        if not point_cloud_path.exists():
            raise FileNotFoundError(f"No matching LiDAR file found for image {image_filename}: {point_cloud_path}")

        return point_cloud_path

    # --- load_matching_point_cloud-------------------------------------------------
    # Takes the image file name and returns lidar points
    def load_matching_point_cloud(self, image_filename: str) -> np.ndarray:
        # Get the lidar path and lidar data from the image file name
        point_cloud_path = self.get_lidar_path_from_image_name(image_filename)
        
        return self.load_point_cloud(point_cloud_path)
    
    # --- get_camera_to_robot_rotation -----------------------------------
    # Fixed rotation to convert from robot frame to OpenCV frame
    def get_camera_to_robot_rotation(self) -> np.ndarray:
        # Rodrigues vecor for -90 degrees about z 
        rot_rod_z = np.array([[0.0], [0.0], [-np.pi / 2.0]])
        # Rotation matrix z
        rot_z, _ = cv2.Rodrigues(rot_rod_z)

        # Rodrigues vecor for -90 degrees about x
        rot_rod_x = np.array([[-np.pi / 2.0], [0.0], [0.0]])
        # Rotation matrix x
        rot_x, _ = cv2.Rodrigues(rot_rod_x)

        # Return the combined rotation matrix
        return rot_z @ rot_x
    
    # --- project_points_to_image ----------------------------------------
    # Project LiDAR points into image pixel coordinates
    def project_points_to_image(
        self,
        lidar_points: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
        
        # Keep only x and y from the LiDAR scan [x, y, z]
        lidar_points_2d = np.asarray(lidar_points[:, :2], dtype=np.float64)

        # Turn into 3D points by adding z = 0
        lidar_points_3d = np.hstack((
            lidar_points_2d,
            np.zeros((lidar_points_2d.shape[0], 1), dtype=np.float64)
        ))

        # Convert planar x, y into homogeneous coordinates
        xy_h = np.hstack((
            lidar_points_3d[:, :2],
            np.ones((lidar_points_3d.shape[0], 1), dtype=np.float64)
        ))

        # Apply the 2D calibration matrix to each LiDAR point to get new x, y 
        camera_robot_xy = (self.lidar_to_camera_2d @ xy_h.T).T[:, :2]

        # Copy 3D points and replace x, y with transformed values
        camera_robot_points = lidar_points_3d.copy()
        camera_robot_points[:, :2] = camera_robot_xy

        # Get the rotation matrix for robot --> opencv 
        camera_to_robot = self.get_camera_to_robot_rotation()
        # Convert from camera robot frame to OpenCV optical frame
        camera_optical_points = (camera_to_robot.T @ camera_robot_points.T).T

        # Clear points not in front of the camera 
        positive_depth_mask = camera_optical_points[:, 2] > 1e-6
        valid_camera_points = camera_optical_points[positive_depth_mask]

        # Return empty if no data avaliable 
        if len(valid_camera_points) == 0:
            return np.empty((0, 2), dtype=np.float64), valid_camera_points

        # Project 3D points into 2D pixel image with openCV and callibration data
        projected_points, _ = cv2.projectPoints(
            valid_camera_points,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            self.camera_matrix,
            self.dist_coeffs,
        )

        # Get the horizontal and vertical pixel position [u, v]
        projected_points = projected_points.reshape(-1, 2)

        # Returns the final 2D pixel locations and matching 3D points after conversion
        return projected_points, valid_camera_points
    
    # --- get_contour_points ------------------------------------------
    # Keep only projected LiDAR points that fall inside a cone contour
    def get_contour_points(
        self,
        projected_points: np.ndarray,
        camera_points: np.ndarray,
        contour: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        # Store projected image points that lie inside the contour
        contour_projected_points = []

        # Store the matching 3D camera-frame points for those image points
        contour_camera_points = []

        # Loop through each projected pixel and its matching 3D point
        for pixel, camera_point in zip(projected_points, camera_points):
            # Convert pixel coordinates to floats for OpenCV
            u = float(pixel[0])
            v = float(pixel[1])

            # Check whether the projected pixel lies inside or on the contour
            is_inside = cv2.pointPolygonTest(contour, (u, v), False)

            # Keep the point if it is inside or on the contour boundary
            if is_inside >= 0:
                contour_projected_points.append(pixel)
                contour_camera_points.append(camera_point)

        # Convert results to numpy arrays
        contour_projected_points = np.array(contour_projected_points, dtype=np.float64)
        contour_camera_points = np.array(contour_camera_points, dtype=np.float64)

        return contour_projected_points, contour_camera_points

    # --- filter_points ----------------------------------------
    # Keep only LiDAR points close to the nearest depth in the contour
    def filter_points(
        self,
        camera_points: np.ndarray,
        threshold_m: float,
    ) -> np.ndarray:

        # Return early if there are no points
        if len(camera_points) == 0:
            return camera_points

        # Compute Euclidean distance of each 3D point from the camera
        distances = np.linalg.norm(camera_points, axis=1)

        # Find the nearest depth
        min_distance = np.min(distances)

        # Keep only points close to the nearest depth
        keep_mask = np.abs(distances - min_distance) <= threshold_m
        filtered_camera_points = camera_points[keep_mask]

        return filtered_camera_points

    # --- get_distance_metrics ----------------------------------------
    # Compute mean and standard deviation of distances for one cone
    def get_distance_metrics(self, camera_points: np.ndarray) -> tuple[float | None, float | None, float | None]:
        # Return None values if no points were matched
        if len(camera_points) == 0:
            return None, None, None

        # Compute Euclidean distance of each 3D point from the camera origin
        distances = np.linalg.norm(camera_points, axis=1)

        # Compute median, mean and standard deviation
        median_distance = float(np.median(distances))
        mean_distance = float(np.mean(distances))
        std_distance = float(np.std(distances))

        return median_distance, mean_distance, std_distance
    
    # --- get_cone_distance_stats ----------------------------------------------
    # Get LiDAR distance statistics for every cone in one image
    def get_cone_distance_stats(
        self,
        cone_detections: list[dict],
        projected_points: np.ndarray,
        valid_camera_points: np.ndarray,
        filter_threshold_m: float,
    ) -> list[dict]:
        cone_stats = []

        # Get cone distances
        for cone_detection in cone_detections:
            contour = cone_detection["contour"]
            cone_id = cone_detection["cone_id"]

            # Get the Lidar points within the countour box
            contour_projected_points, contour_camera_points = self.get_contour_points(
                projected_points,
                valid_camera_points,
                contour
            )

            # Filter obvious distance outliers for this cone
            filtered_camera_points = self.filter_points(
                contour_camera_points,
                filter_threshold_m
            )
        
            # Get median, mean and standard deviation after filtering
            median_distance, mean_distance, std_distance = self.get_distance_metrics(
                filtered_camera_points
            )

            cone_stats.append(
                {
                    "cone_id": cone_id,
                    "center_point": cone_detection["center_point"],
                    "contour": contour,
                    "contour_projected_points": contour_projected_points,
                    "median_distance": median_distance,
                    "mean_distance": mean_distance,
                    "std_distance": std_distance,
                }
            )

        return cone_stats
    
    # --- draw_projected_points ------------------------------------------
    # Draw projected LiDAR points on the image
    def draw_projected_points(
        self,
        image_bgr: np.ndarray,
        projected_points: np.ndarray,
    ) -> np.ndarray:
        
        overlay = image_bgr.copy()
        image_height, image_width = overlay.shape[:2]

        for pixel in projected_points:
            u = int(round(pixel[0]))
            v = int(round(pixel[1]))

            if 0 <= u < image_width and 0 <= v < image_height:
                cv2.circle(overlay, (u, v), 3, (0, 0, 255), -1)

        return overlay
    
    # --- save_lidar_distance_image ---------------------------------------------
    # Save image showing cone contours, contour LiDAR points and distance labels
    def save_lidar_distance_image(
        self,
        output_path: str | Path,
        image_bgr: np.ndarray,
        cone_stats: list[dict],
    ) -> None:
        contour_debug_image = image_bgr.copy()

        # Draw debug information for each cone
        for cone_stat in cone_stats:
            contour = cone_stat["contour"]
            cone_id = cone_stat["cone_id"]
            center_point = cone_stat["center_point"]
            contour_projected_points = cone_stat["contour_projected_points"]
            mean_distance = cone_stat["mean_distance"]
            std_distance = cone_stat["std_distance"]

            # Draw contour
            cv2.drawContours(contour_debug_image, [contour], -1, (0, 255, 0), 2)

            # Draw projected LiDAR points inside this contour
            for pixel in contour_projected_points:
                u = int(round(pixel[0]))
                v = int(round(pixel[1]))
                cv2.circle(contour_debug_image, (u, v), 4, (0, 0, 255), -1)

            # Label each cone with mean distance and standard deviation
            if mean_distance is None:
                label = f"Cone {cone_id}: no LiDAR"
            else:
                label = f"Cone {cone_id}: {mean_distance:.2f}m  {std_distance:.2f}"

            cv2.putText(
                contour_debug_image,
                label,
                (center_point[0], center_point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        output_path = Path(output_path)
        cv2.imwrite(str(output_path), contour_debug_image)

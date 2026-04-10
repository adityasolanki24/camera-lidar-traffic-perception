from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


DATASET_PATH = Path("extracted_traffic_cone_images")
HSV_OUTPUT_PATH = Path("first_image_hsv_plot.png")


def convert_bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
def save_hsv_diagnostic_plot(image_bgr: np.ndarray, output_path: Path) -> None:
    image_rgb = convert_bgr_to_rgb(image_bgr)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv_pixels = image_hsv.reshape(-1, 3)
    rgb_pixels = image_rgb.reshape(-1, 3) / 255.0

    max_points = 12000
    if len(hsv_pixels) > max_points:
        sample_indices = np.linspace(0, len(hsv_pixels) - 1, max_points, dtype=int)
        hsv_pixels = hsv_pixels[sample_indices]
        rgb_pixels = rgb_pixels[sample_indices]

    hue_channel = hsv_pixels[:, 0]
    saturation_channel = hsv_pixels[:, 1]
    value_channel = hsv_pixels[:, 2]

    figure = plt.figure(figsize=(14, 6))
    axis_image = figure.add_subplot(1, 2, 1)
    axis_hsv = figure.add_subplot(1, 2, 2, projection="3d")

    axis_image.imshow(image_rgb)
    axis_image.set_title("First image")
    axis_image.axis("off")

    axis_hsv.scatter(
        hue_channel,
        saturation_channel,
        value_channel,
        c=rgb_pixels,
        s=4,
        alpha=0.6,
        linewidths=0,
    )
    axis_hsv.set_title("HSV colour space")
    axis_hsv.set_xlabel("Hue")
    axis_hsv.set_ylabel("Saturation")
    axis_hsv.set_zlabel("Value")
    axis_hsv.set_xlim(0, 180)
    axis_hsv.set_ylim(0, 255)
    axis_hsv.set_zlim(0, 255)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved 3D HSV plot to: {output_path}")
    print(
        "First image HSV ranges:"
        f" H=[{int(image_hsv[:, :, 0].min())}, {int(image_hsv[:, :, 0].max())}]"
        f" S=[{int(image_hsv[:, :, 1].min())}, {int(image_hsv[:, :, 1].max())}]"
        f" V=[{int(image_hsv[:, :, 2].min())}, {int(image_hsv[:, :, 2].max())}]"
    )


def main() -> None:
    dataset_path = DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")

    first_image = None
    for file_path in dataset_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() == ".png":
            first_image = cv2.imread(str(file_path))
            if first_image is None:
                raise ValueError(f"Failed to read image: {file_path}")
            break

    if first_image is None:
        raise FileNotFoundError(f"No .png images found in dataset: {dataset_path}")

    save_hsv_diagnostic_plot(first_image, HSV_OUTPUT_PATH)


if __name__ == "__main__":
    main()

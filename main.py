import cv2
import numpy as np
from PIL import Image
from typing import Tuple

class PencilMeasurement:
    def __init__(self, image_path: str, width_in_inches: float = None, height_in_inches: float = None):
        """
        Initialize the PencilMeasurement with the given image path and optionally, the physical size.

        :param image_path: Path to the image file.
        :param width_in_inches: Optional physical width of the image in inches.
        :param height_in_inches: Optional physical height of the image in inches.
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.denoised_image = None
        self.bbox_a = None
        self.bbox_b = None
        self.dpi = self.calculate_dpi(width_in_inches, height_in_inches)
        # self.conversion_factor = 25.4 / 96  # Conversion factor from pixels to millimeters
        self.conversion_factor = 25.4 / self.dpi  # Conversion factor from pixels to millimeters

        print(f"Calculated DPI: {self.dpi}")
        print(f"Conversion Factor (pixels to mm): {self.conversion_factor}")

    def calculate_dpi(self, width_in_inches: float = None, height_in_inches: float = None) -> int:
        """
        Calculate the DPI (dots per inch) of the image.

        :param width_in_inches: Optional physical width of the image in inches.
        :param height_in_inches: Optional physical height of the image in inches.
        :return: The calculated or inferred DPI.
        """
        # Try to read DPI from the image metadata
        with Image.open(self.image_path) as img:
            dpi_info = img.info.get('dpi')
            if dpi_info:
                return int(dpi_info[0])  # DPI is usually given as a tuple (x_dpi, y_dpi)

            # If width and height in inches are provided, calculate DPI
            if width_in_inches and height_in_inches:
                width_pixels, height_pixels = img.size
                dpi_width = width_pixels / width_in_inches
                dpi_height = height_pixels / height_in_inches
                return int((dpi_width + dpi_height) / 2)  # Average DPI

        # Default DPI if none of the above methods work
        return 300  # Common default DPI

    def denoise_image(self) -> None:
        """Apply Non-Local Means Denoising to the image."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.denoised_image = cv2.fastNlMeansDenoising(gray, None, 48, 10, 21)

    def select_bounding_boxes(self) -> None:
        """Allow the user to select bounding boxes for two pencils."""
        print("Select the bounding box for Pencil-A.")
        self.bbox_a = cv2.selectROI("Select Pencil-A on Denoised Image", self.denoised_image, fromCenter=False, showCrosshair=True)

        print("Select the bounding box for Pencil-B.")
        self.bbox_b = cv2.selectROI("Select Pencil-B on Denoised Image", self.denoised_image, fromCenter=False, showCrosshair=True)

    def calculate_diagonals(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int, float]:
        """
        Calculate the diagonal coordinates and length of a bounding box.

        :param bbox: The bounding box coordinates (x, y, width, height).
        :return: Coordinates of the diagonal points and the diagonal length in pixels.
        """
        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        length = np.sqrt((x2 - x) ** 2 + (y2 - y) ** 2)
        return x, y, x2, y2, length

    def draw_diagonals_and_measurements(self) -> None:
        """Draw the diagonals of the selected bounding boxes and measure lengths and angles."""
        if self.bbox_a != (0, 0, 0, 0) and self.bbox_b != (0, 0, 0, 0):
            x1_a, y1_a, x2_a, y2_a, length_a_px = self.calculate_diagonals(self.bbox_a)
            x1_b, y1_b, x2_b, y2_b, length_b_px = self.calculate_diagonals(self.bbox_b)

            length_a_mm = length_a_px * self.conversion_factor
            length_b_mm = length_b_px * self.conversion_factor

            print(f"Length of Pencil-A in pixels: {length_a_px}")
            print(f"Length of Pencil-B in pixels: {length_b_px}")
            # print(f"Length of Pencil-A in mm: {length_a_mm}")
            # print(f"Length of Pencil-B in mm: {length_b_mm}")

            diagonal_vector_a = np.array([x2_a - x1_a, y2_a - y1_a])
            diagonal_vector_b = np.array([x2_b - x1_b, y2_b - y1_b])

            dot_product = np.dot(diagonal_vector_a, diagonal_vector_b)
            magnitude_a = np.linalg.norm(diagonal_vector_a)
            magnitude_b = np.linalg.norm(diagonal_vector_b)

            angle_rad = np.arccos(dot_product / (magnitude_a * magnitude_b))
            angle_deg = np.degrees(angle_rad)

            intersection_x = (x1_a + x2_a + x1_b + x2_b) // 4
            intersection_y = (y1_a + y2_a + y1_b + y2_b) // 4

            image_with_diagonals = cv2.cvtColor(self.denoised_image, cv2.COLOR_GRAY2BGR)
            cv2.line(image_with_diagonals, (x1_a, y1_a), (x2_a, y2_a), (0, 0, 0), 2)
            cv2.line(image_with_diagonals, (x1_b, y1_b), (x2_b, y2_b), (0, 255, 0), 2)
            cv2.ellipse(image_with_diagonals, (intersection_x, intersection_y), (30, 30), 0, 0, angle_deg, (0, 0, 255), 2)

            cv2.putText(image_with_diagonals, f"Length A: {length_a_mm:.2f} mm", (x1_a, y1_a - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(image_with_diagonals, f"Length B: {length_b_mm:.2f} mm", (x1_b, y1_b - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image_with_diagonals, f"Angle: {angle_deg:.2f} degrees", (intersection_x + 40, intersection_y + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.namedWindow('Pencils with Diagonals', cv2.WINDOW_NORMAL)
            cv2.imshow('Pencils with Diagonals', image_with_diagonals)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print(f"Length of Pencil-A: {length_a_mm:.2f} mm")
            print(f"Length of Pencil-B: {length_b_mm:.2f} mm")
            print(f"Angle between Pencils: {angle_deg:.2f} degrees")
        else:
            print("Bounding boxes were not properly selected.")

    def process(self) -> None:
        """Main method to run the pencil measurement process."""
        self.denoise_image()
        self.select_bounding_boxes()
        self.draw_diagonals_and_measurements()

if __name__ == "__main__":
    # Replace 'pencils.jpg' with your actual image path and physical dimensions if known
    pencil_measurement = PencilMeasurement('pencils.jpg')
    pencil_measurement.process()

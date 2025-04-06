#%%
import cv2
import numpy as np
import gdspy

def extract_polygon_and_write_gds(
    image_path,
    output_gds_path,
    threshold_value=200,
    min_contour_area=1000
):
    """
    Reads an image, detects the largest bright region (white shape),
    converts it to a polygon, and writes it to a GDSII file.
    
    :param image_path: Path to the input image.
    :param output_gds_path: Path for the output GDS file.
    :param threshold_value: Intensity threshold for binarization.
    :param min_contour_area: Ignore contours smaller than this area.
    """

    # 1. Read image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise IOError(f"Could not read image: {image_path}")

    # 2. Threshold to isolate bright/white areas
    # You may want to adjust threshold_value or use cv2.THRESH_OTSU for automatic threshold
    _, thresh = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Optional: remove small noise with morphological opening or closing
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 3. Find external contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Filter and pick the contour that presumably corresponds to the central shape
    #    (Here, we simply pick the largest contour above a minimum area.)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    if not valid_contours:
        raise ValueError("No contours found with area > min_contour_area.")

    # Sort contours by area (descending)
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    main_contour = valid_contours[0]

    # 5. Optionally approximate the contour to reduce number of points
    #    Epsilon is a parameter: the smaller it is, the closer to the original contour
    epsilon = 1.0
    approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)

    # Convert to a list of (x, y) tuples
    points = [(pt[0][0], pt[0][1]) for pt in approx_contour]

    # 6. Create a GDSII library and cell
    gds_lib = gdspy.GdsLibrary()
    cell = gds_lib.new_cell("MAIN_CELL")

    # Create a polygon on layer 0 (change layer as needed)
    polygon = gdspy.Polygon(points, layer=0)
    cell.add(polygon)

    # 7. Write to GDS file
    gds_lib.write_gds(output_gds_path)
    print(f"GDSII file successfully written to {output_gds_path}")

#%%

# if __name__ == "__main__":
# Example usage
image_file = "test2.png"
gds_output_file = "output.gds"

extract_polygon_and_write_gds(
    image_path=image_file,
    output_gds_path=gds_output_file,
    threshold_value=200,
    min_contour_area=500  # Adjust based on your image scale
)

# %%

#%%
import cv2
import numpy as np
import gdspy
import matplotlib.pyplot as plt

def extract_polygon_and_write_gds(
    image_path,
    output_gds_path,
    threshold_value=200,
    min_contour_area=1000,
    debug=False
):
    """
    Reads an image, detects the largest bright region (white shape),
    converts it to a polygon, and writes it to a GDSII file.
    
    Also prints and plots debug information if debug is True.
    
    :param image_path: Path to the input image.
    :param output_gds_path: Path for the output GDS file.
    :param threshold_value: Intensity threshold for binarization.
    :param min_contour_area: Ignore contours smaller than this area.
    :param debug: If True, prints debug information and plots intermediate steps.
    """

    # 1. Read image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise IOError(f"Could not read image: {image_path}")
    
    if debug:
        print(f"[DEBUG] Loaded image shape: {image_gray.shape}")
        plt.figure()
        plt.imshow(image_gray, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')
        plt.show()

    # 2. Threshold to isolate bright/white areas
    ret, thresh = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)
    if debug:
        print(f"[DEBUG] Threshold value used: {threshold_value}")
        print(f"[DEBUG] Number of white pixels after threshold: {np.sum(thresh==255)}")
        plt.figure()
        plt.imshow(thresh, cmap='gray')
        plt.title("Thresholded Image")
        plt.axis('off')
        plt.show()
    
    # Optional: remove small noise with morphological operations
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 3. Find external contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        print(f"[DEBUG] Total contours found: {len(contours)}")
    
    # 4. Filter and pick the contour that presumably corresponds to the central shape
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    if debug:
        print(f"[DEBUG] Contours with area greater than {min_contour_area}: {len(valid_contours)}")
    
    if not valid_contours:
        raise ValueError("No contours found with area > min_contour_area.")
    
    # Sort contours by area (descending)
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    main_contour = valid_contours[0]
    area = cv2.contourArea(main_contour)
    if debug:
        print(f"[DEBUG] Selected main contour area: {area}")
        
        # Draw all valid contours on a copy of the grayscale image for visualization
        image_with_contours = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_with_contours, valid_contours, -1, (0, 255, 0), 2)
        # Convert BGR to RGB for correct display with matplotlib
        image_with_contours = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(image_with_contours)
        plt.title("Valid Contours on Grayscale Image")
        plt.axis('off')
        plt.show()

    # 5. Optionally approximate the contour to reduce number of points
    epsilon = 1.0  # Adjust epsilon as needed
    approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
    if debug:
        print(f"[DEBUG] Approximated polygon vertices count: {len(approx_contour)}")
    
    # Convert to a list of (x, y) tuples
    points = [(pt[0][0], pt[0][1]) for pt in approx_contour]
    if debug:
        print(f"[DEBUG] Polygon points: {points}")
        
        # Plot the approximated polygon overlaid on the grayscale image
        plt.figure()
        plt.imshow(image_gray, cmap='gray')
        # Unzip points to x and y lists and close the polygon by repeating the first point
        if points:
            x_points, y_points = zip(*points)
            x_points = list(x_points) + [x_points[0]]
            y_points = list(y_points) + [y_points[0]]
            plt.plot(x_points, y_points, marker='o', linestyle='-', label='Approximated Polygon')
        plt.title("Detected Polygon on Grayscale Image")
        plt.legend()
        plt.axis('off')
        plt.show()

    # 6. Create a GDSII library and cell
    gds_lib = gdspy.GdsLibrary()
    cell = gds_lib.new_cell("MAIN_CELL")
    polygon = gdspy.Polygon(points, layer=0)
    cell.add(polygon)
    
    if debug:
        print("[DEBUG] Polygon added to GDS cell.")

    # 7. Write to GDS file
    gds_lib.write_gds(output_gds_path)
    if debug:
        print(f"[DEBUG] GDSII file successfully written to {output_gds_path}")

#%%

# if __name__ == "__main__":
# Example usage
image_file = "test2.png"
gds_output_file = "output.gds"

extract_polygon_and_write_gds(
    image_path=image_file,
    output_gds_path=gds_output_file,
    threshold_value=120,
    min_contour_area=500,  # Adjust based on your image scale
    debug=True
)

# %%

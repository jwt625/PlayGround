
#%%
import cv2
import numpy as np
import os

#%%
def process_thermal_image(image_path, min_temp, max_temp):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Isolate the central white square
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the white square)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Create a mask for the square
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)

    # Calculate average intensity of the square
    square_region = cv2.bitwise_and(gray, gray, mask=mask)
    non_zero_pixels = square_region[np.nonzero(square_region)]
    avg_intensity = np.mean(non_zero_pixels)

    # Map average intensity to temperature
    temp_range = max_temp - min_temp
    avg_temp = min_temp + (avg_intensity / 255.0) * temp_range

    # Visualize the result
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img, avg_temp

# Folder containing the images
image_folder = 'images'

# Get list of image files sorted by name
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# User-provided temperature ranges (example, replace with actual values)
min_temps = [1.7, 1.6, 2.1, 2.4, 2.5, 3.2, 4.2, 4.3, 4.7, 4.9, 5.4, 5.7, 6.4, 6.9, 7.2, 7.2, 7.5, 7.7, 8.2, 8.4, 8.6, 9.2, 9.5, 9.6, ]  # Replace with actual minimum temperatures
max_temps = [13.1, 12.4, 12.7, 12.2, 12.5, 13.1, 13.3, 12.5, 13.1, 13.2, 14.7, 13.8, 13.5, 15.3, 15.1, 15.2, 16.3, 16.5, 16.4, 17.3, 18.0, 17.7, 17.7, 17.9]  # Replace with actual maximum temperatures

# Ensure the number of temperature ranges matches the number of images
assert len(image_files) == len(min_temps) == len(max_temps), "Number of images and temperature ranges must match"

# Process each image
T_mins = []
T_maxs = []
all_avg_temp = []
for i, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)
    min_temp = min_temps[i]
    max_temp = max_temps[i]

    result_img, avg_temp = process_thermal_image(image_path, min_temp, max_temp)

    print(f"Image: {image_file}")
    print(f"Minimum temperature: {min_temp:.1f}°C")
    print(f"Maximum temperature: {max_temp:.1f}°C")
    print(f"Average temperature of the square: {avg_temp:.1f}°C")
    T_mins.append(min_temp)
    T_maxs.append(max_temp)
    all_avg_temp.append(avg_temp)
    print()

    # Save the result
    result_path = os.path.join('results', f'result_{image_file}')
    os.makedirs('results', exist_ok=True)
    cv2.imwrite(result_path, result_img)

print("All images processed. Results saved in the 'results' folder.")
# %%
import matplotlib.pyplot  as plt

plt.plot(all_avg_temp)
plt.ylabel('Temperature (C)')
# %%

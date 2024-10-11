
#%%
import cv2
import numpy as np

#%%
# Load the image
img = cv2.imread('./images/IMG20241010231524.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Detect temperature range from color bar
color_bar = gray[:, -20:]  # Assume color bar is on the right side
min_temp = np.min(color_bar)
max_temp = np.max(color_bar)

# Step 2: Isolate the central white square
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming it's the white square)
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)

# Create a mask for the square
mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [largest_contour], 0, 255, -1)

# Step 3: Calculate average temperature of the square
square_region = cv2.bitwise_and(gray, gray, mask=mask)
non_zero_pixels = square_region[np.nonzero(square_region)]
avg_intensity = np.mean(non_zero_pixels)

# Map average intensity to temperature
temp_range = max_temp - min_temp
avg_temp = min_temp + (avg_intensity / 255.0) * temp_range

print(f"Minimum temperature: {min_temp:.1f}")
print(f"Maximum temperature: {max_temp:.1f}")
print(f"Average temperature of the square: {avg_temp:.1f}")

# Visualize the result
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Save the result instead of displaying it
cv2.imwrite('result.png', img)

print("Result image saved as 'result.png'")
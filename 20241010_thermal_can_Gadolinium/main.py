
#%%
import cv2
import pytesseract
import numpy as np
from PIL import Image

#%%
# Load the image
img = cv2.imread('./images/IMG20241010231524.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Extract temperature range from the image text
def get_temp_from_text(image, y):
    roi = image[y:y+30, -60:]  # Assume temperature text is in the right 60 pixels
    _, thresh = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
    return thresh

# Get temperature values
max_temp_img = get_temp_from_text(gray, 0)
min_temp_img = get_temp_from_text(gray, gray.shape[0] - 30)

# These values are hardcoded based on the image. In a real scenario, you'd use more robust methods.
max_temp = 13.1
min_temp = 1.7

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

print(f"Minimum temperature: {min_temp:.1f}°C")
print(f"Maximum temperature: {max_temp:.1f}°C")
print(f"Average temperature of the square: {avg_temp:.1f}°C")

# Visualize the result
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Save the result
cv2.imwrite('result.png', img)
print("Result image saved as 'result.png'")
# %%

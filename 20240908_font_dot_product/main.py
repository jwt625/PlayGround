#%% 
# pip install numpy pillow



#%%
# import numpy


#%%
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Function to render a character from a font into an image
def render_letter(letter, font_path, image_size=(100, 100), font_size=80):
    font = ImageFont.truetype(font_path, font_size)
    image = Image.new('L', image_size, color=255)  # 'L' mode for grayscale
    draw = ImageDraw.Draw(image)
    w, h = draw.textsize(letter, font=font)
    draw.text(((image_size[0] - w) / 2, (image_size[1] - h) / 2), letter, font=font, fill=0)
    return np.array(image)

# Function to calculate dot product of two images (flattened arrays)
def dot_product(img1, img2):
    return np.dot(img1.flatten(), img2.flatten())

# Load two different fonts (download .ttf fonts and provide the path)
font1_path = "Times New Roman.ttf"
font2_path = "helvetica.ttf"

# Choose a letter
letter = "A"

# Render the letter in both fonts
img1 = render_letter(letter, font1_path)
img2 = render_letter(letter, font2_path)

# Calculate the dot product of the two letter images
dot_prod = dot_product(img1, img2)

print(f"Dot product for letter '{letter}' between two fonts: {dot_prod}")

# Optionally, display the images
Image.fromarray(img1).show()
Image.fromarray(img2).show()

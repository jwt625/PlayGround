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
    
    # Get text bounding box (coordinates of the rendered text)
    bbox = draw.textbbox((0, 0), letter, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Draw the text centered
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



#%%
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import string
import matplotlib.pyplot as plt

# Function to render a character from a font into an image
def render_letter(letter, font_path, image_size=(100, 100), font_size=80):
    font = ImageFont.truetype(font_path, font_size)
    image = Image.new('L', image_size, color=255)  # 'L' mode for grayscale
    draw = ImageDraw.Draw(image)
    
    # Get text bounding box (coordinates of the rendered text)
    bbox = draw.textbbox((0, 0), letter, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Draw the text centered
    draw.text(((image_size[0] - w) / 2, (image_size[1] - h) / 2), letter, font=font, fill=0)
    return np.array(image)

# Function to calculate dot product of two images (flattened arrays)
def dot_product(img1, img2):
    return np.dot(img1.flatten(), img2.flatten())

# Function to compute dot products for all letters in a font
def compute_dot_matrix(font_path):
    alphabet = string.ascii_uppercase  # List of all uppercase letters
    n = len(alphabet)
    
    # Initialize an empty matrix to store dot products
    dot_matrix = np.zeros((n, n))
    
    # Render all letters from the font
    images = {letter: render_letter(letter, font_path) for letter in alphabet}
    
    # Compute dot products between all pairs of letters
    for i, letter1 in enumerate(alphabet):
        for j, letter2 in enumerate(alphabet):
            img1 = 1.0 - images[letter1] / 255.0
            img2 = 1.0 - images[letter2] / 255.0
            d1 = dot_product(img1, img1)
            d2 = dot_product(img2, img2)
            dot_matrix[i, j] = dot_product(img1, img2)/np.sqrt(d1*d2)
    
    return dot_matrix

# Function to plot the dot product matrix
def plot_dot_matrix(dot_matrix, str_font):
    plt.figure(figsize=(8, 8))
    plt.imshow(dot_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Dot Product")
    plt.xticks(ticks=np.arange(26), labels=list(string.ascii_uppercase))
    plt.yticks(ticks=np.arange(26), labels=list(string.ascii_uppercase))
    plt.title(f'Dot Product Matrix of Letters, {str_font}')
    plt.show()

# Load the font and compute the dot product matrix
# font_path = "Times New Roman.ttf"
# font_path = "helvetica.ttf"
# font_path = "JetBrainsMono-Medium.ttf"
font_path = "Comic Sans MS.ttf"
dot_matrix = compute_dot_matrix(font_path)

# Plot the dot product matrix
plot_dot_matrix(dot_matrix, font_path)
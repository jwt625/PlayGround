
#%%
import sys
from PIL import Image


#%%
def reduce_colors(pixel, bits_per_channel=4):
    r, g, b = pixel
    r = round(r * (2**bits_per_channel - 1) / 255) * (255 // (2**bits_per_channel - 1))
    g = round(g * (2**bits_per_channel - 1) / 255) * (255 // (2**bits_per_channel - 1))
    b = round(b * (2**bits_per_channel - 1) / 255) * (255 // (2**bits_per_channel - 1))
    return (r, g, b)

def convert_to_pixel_art(input_path, output_path, new_size=(320, 180), 
                         bits_per_channel=4):
    with Image.open(input_path) as img:
        # Resize the image
        img = img.resize(new_size, Image.NEAREST)
        
        # Convert to RGB mode if it's not already
        img = img.convert('RGB')
        
        # Reduce colors
        pixels = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                pixels[i, j] = reduce_colors(pixels[i, j], bits_per_channel)
        
        # Save the result
        img.save(output_path)

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python script.py <input_image> <output_image>")
#         sys.exit(1)
    
#     input_path = sys.argv[1]
#     output_path = sys.argv[2]
    
#     convert_to_pixel_art(input_path, output_path)
#     print(f"Pixel art image saved as {output_path}")

#%%
# input_path = "C:/Users/Wentao/Desktop/photos/unnamed.png"
# input_path = "C:/Users/Wentao/Desktop/photos/IMG_9200.JPG"
# input_path = "OHMP.PNG"
input_path = "OHMP_process.PNG"
output_path = "output.png"
# size = (480, 270)
size = (640, 270)

convert_to_pixel_art(input_path, output_path, new_size=size, bits_per_channel=2)
print(f"Pixel art image saved as {output_path}")



#%% second version with Floyd–Steinberg dithering
import sys
import numpy as np
from PIL import Image

def quantize_color(color, palette):
    return palette[np.argmin(np.sum((palette[:, np.newaxis] - color) ** 2, axis=2))]

def floyd_steinberg_dither(image, palette):
    height, width = image.shape[:2]
    new_img = np.zeros_like(image)
    
    for y in range(height):
        for x in range(width):
            old_pixel = image[y, x].astype(float)
            new_pixel = quantize_color(old_pixel, palette)
            new_img[y, x] = new_pixel
            
            quant_error = old_pixel - new_pixel
            
            if x + 1 < width:
                image[y, x + 1] = np.clip(image[y, x + 1] + quant_error * 7/16, 0, 255)
            if x - 1 >= 0 and y + 1 < height:
                image[y + 1, x - 1] = np.clip(image[y + 1, x - 1] + quant_error * 3/16, 0, 255)
            if y + 1 < height:
                image[y + 1, x] = np.clip(image[y + 1, x] + quant_error * 5/16, 0, 255)
            if x + 1 < width and y + 1 < height:
                image[y + 1, x + 1] = np.clip(image[y + 1, x + 1] + quant_error * 1/16, 0, 255)
    
    return new_img

def create_palette(bits_per_channel):
    values = np.linspace(0, 255, 2**bits_per_channel).astype(int)
    return np.array(np.meshgrid(values, values, values)).T.reshape(-1, 3)

def convert_to_pixel_art(input_path, output_path, new_size=(320, 180), bits_per_channel=2):
    with Image.open(input_path) as img:
        # Resize the image
        img = img.resize(new_size, Image.LANCZOS)
        
        # Convert to RGB mode if it's not already
        img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Create color palette
        palette = create_palette(bits_per_channel)
        
        # Apply dithering
        dithered = floyd_steinberg_dither(img_array, palette)
        
        # Convert back to PIL Image and save
        result = Image.fromarray(dithered.astype('uint8'))
        result.save(output_path)

# if __name__ == "__main__":
#     if len(sys.argv) != 4:
#         print("Usage: python script.py <input_image> <output_image> <bits_per_channel>")
#         sys.exit(1)
    
#     input_path = sys.argv[1]
#     output_path = sys.argv[2]
#     bits_per_channel = int(sys.argv[3])
    

#%%

# input_path = "C:/Users/Wentao/Desktop/photos/unnamed.png"
# input_path = "C:/Users/Wentao/Desktop/photos/IMG_9200.JPG"
# input_path = "C:/Users/Wentao/Desktop/photos/research/IMG_8775.JPG"
# input_path = "C:/Users/Wentao/Desktop/photos/research/IMG_6329.PNG"
input_path = "C:/Users/Wentao/Desktop/photos/research/nanobeam_ship.png"


# input_path = "OHMP.PNG"
# input_path = "OHMP_process.PNG"
# output_path = "output.png"
# output_path = "output2.png"
# output_path = "output3.png"
# output_path = "output4.png"
output_path = "output5.png"


# size = (480, 270)
# size = (360, 200)
size = (360, 240)
# size = (320, 130)

convert_to_pixel_art(input_path, output_path, new_size=size, bits_per_channel=2)
print(f"Pixel art image saved as {output_path}")




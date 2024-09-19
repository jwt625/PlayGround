
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






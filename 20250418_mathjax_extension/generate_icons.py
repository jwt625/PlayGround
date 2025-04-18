

#%%

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size):
    # Create a new image with a blue background
    image = Image.new('RGB', (size, size), '#2196F3')
    draw = ImageDraw.Draw(image)
    
    # Try to use Arial font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial", size=int(size * 0.5))
    except:
        font = ImageFont.load_default()
    
    # Draw the sigma symbol
    text = "∑"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    draw.text((x, y), text, fill='white', font=font)
    
    # Save the image
    image.save(f'icon{size}.png')

# Generate both icon sizes
create_icon(48)
create_icon(128) 
# %%

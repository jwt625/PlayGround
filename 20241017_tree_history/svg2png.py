
#%%
from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, color, text_color, output_path):
    """
    Create a square icon with the letter "T" in the center, slightly raised
    
    Args:
        size (int): Size of the icon in pixels
        color (str): Background color in hex format
        text_color (str): Text color in hex format
        output_path (str): Path to save the PNG file
    """
    # Create a new image with the specified background color
    image = Image.new('RGB', (size, size), color)
    draw = ImageDraw.Draw(image)
    
    # Calculate font size (proportional to image size)
    font_size = int(size * 0.7)  # 70% of icon size
    
    try:
        # Try to use Arial font
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            # Fallback to Times New Roman
            font = ImageFont.truetype("times.ttf", font_size)
        except OSError:
            # Final fallback to default font
            font = ImageFont.load_default()
            font_size = size // 2  # Adjust size for default font
    
    # Get the size of the text
    text = "T"
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top
    except AttributeError:  # For older Pillow versions
        text_width, text_height = draw.textsize(text, font=font)
    
    # Calculate position to center the text horizontally and raise it vertically
    x = (size - text_width) // 2
    # Move text up by 15% of the icon size
    vertical_offset = size * 0.15
    y = ((size - text_height) // 2) - vertical_offset
    
    # Draw the text
    draw.text((x, y), text, font=font, fill=text_color)
    
    # Save the image
    image.save(output_path)
    print(f"Generated: {output_path}")

def main():
    # Create icons directory if it doesn't exist
    os.makedirs('icons', exist_ok=True)
    
    # Define colors
    active_bg = "#90EE90"  # Light green
    active_text = "#333333"  # Dark gray
    inactive_bg = "#D3D3D3"  # Light gray
    inactive_text = "#666666"  # Medium gray
    
    # Sizes to generate
    sizes = (16, 32, 48, 128)
    
    # Generate all icons
    for size in sizes:
        # Active icons
        create_icon(
            size=size,
            color=active_bg,
            text_color=active_text,
            output_path=f'icons/active_{size}.png'
        )
        
        # Inactive icons
        create_icon(
            size=size,
            color=inactive_bg,
            text_color=inactive_text,
            output_path=f'icons/inactive_{size}.png'
        )
    
    print("\nAll icons generated successfully!")
    print("Place the 'icons' folder in your extension directory.")

#%%
main()
# %%

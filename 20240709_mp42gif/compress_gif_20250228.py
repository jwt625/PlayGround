import imageio
import os
from PIL import Image
import sys
import tempfile
import shutil

def compress_gif(input_path, output_path, target_size_mb=15):
    """
    Compress a GIF file to be smaller than the target size in MB.
    
    Args:
        input_path (str): Path to the input GIF file
        output_path (str): Path where the compressed GIF will be saved
        target_size_mb (float): Target size in MB (default 15MB)
    """
    # Get the current file size
    current_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Original GIF size: {current_size_mb:.2f} MB")
    
    if current_size_mb <= target_size_mb:
        print("File is already smaller than target size. No compression needed.")
        shutil.copy(input_path, output_path)
        return
    
    # Calculate compression ratio needed
    compression_ratio = target_size_mb / current_size_mb
    
    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Read the GIF
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data().get('fps', 10)  # Default to 10 FPS if not specified
        
        # Extract original dimensions
        first_frame = reader.get_next_data()
        original_width, original_height = first_frame.shape[1], first_frame.shape[0]
        
        # Calculate new dimensions (maintain aspect ratio)
        # Start with 80% of the original size and adjust as needed
        scale_factor = 0.8
        
        # Try different compression strategies until we get below target size
        quality = 85  # Start with good quality
        
        while True:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            frames = []
            reader = imageio.get_reader(input_path)
            
            for i, frame in enumerate(reader):
                pil_img = Image.fromarray(frame)
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                
                # Save the frame to a temp file to apply quality reduction
                temp_frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                pil_img.save(temp_frame_path, quality=quality)
                
                # Read it back
                frames.append(imageio.imread(temp_frame_path))
            
            # Write the output GIF
            imageio.mimsave(output_path, frames, fps=fps)
            
            # Check if we achieved the target size
            new_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Compressed GIF size with scale={scale_factor:.2f}, quality={quality}: {new_size_mb:.2f} MB")
            
            if new_size_mb <= target_size_mb:
                print(f"Successfully compressed GIF to {new_size_mb:.2f} MB")
                break
            
            # Adjust parameters for next attempt
            if quality > 60:
                # First try reducing quality
                quality -= 10
            else:
                # Then try reducing size
                scale_factor *= 0.9
            
            # Safety check to prevent infinite loop
            if scale_factor < 0.2:
                print("Warning: Reached minimum scale factor. Using best result so far.")
                break
    
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compress_gif.py input.gif output.gif")
        sys.exit(1)
    
    input_gif = sys.argv[1]
    output_gif = sys.argv[2]
    
    if not os.path.exists(input_gif):
        print(f"Error: Input file '{input_gif}' not found.")
        sys.exit(1)
    
    compress_gif(input_gif, output_gif)
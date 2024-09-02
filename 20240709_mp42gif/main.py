#%%
# import numpy
#%%

from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(mp4_path, gif_path, start_time=0, end_time=None):
    # Load the video
    video_clip = VideoFileClip(mp4_path)
    
    # If end_time is not specified, use the full video duration
    if end_time is None:
        end_time = video_clip.duration
    
    # Trim the video to the desired segment
    trimmed_clip = video_clip.subclip(start_time, end_time)
    
    # Write the result to a GIF file
    trimmed_clip.write_gif(gif_path, fps=10)  # You can adjust fps as needed

# #%% Example usage
# mp4_path = "Animation_Different Quantities Animation.mp4"
# gif_path = "Animation_Different Quantities Animation.gif"
# mp4_path = "This tiny solar-powered flyer weighs less than a paper plane - YouTube - Google Chrome 2024-07-18 20-59-53.mp4"
# gif_path = "This tiny solar-powered flyer weighs less than a paper plane - YouTube - Google Chrome 2024-07-18 20-59-53.gif"
# mp4_path = "This tiny solar-powered flyer weighs less than a paper plane - YouTube - Google Chrome 2024-07-18 21-00-24.mp4"
# gif_path = "This tiny solar-powered flyer weighs less than a paper plane - YouTube - Google Chrome 2024-07-18 21-00-24.gif"
# mp4_path = "DTI SmartBolts_ Demonstration - YouTube - Google Chrome 2024-07-20 22-32-12.mp4"
# gif_path = "DTI SmartBolts_ Demonstration - YouTube - Google Chrome 2024-07-20 22-32-12.gif"
mp4_path = "Minecraft_update_Google Chrome 2024-08-24 10-44-35.mp4"
gif_path = "Minecraft_update_Google Chrome 2024-08-24 10-44-35.gif"


convert_mp4_to_gif(mp4_path, gif_path)

# %% compress the gif
from PIL import Image

def compress_gif(input_path, output_path, quality=50):
    # Open the GIF file
    gif = Image.open(input_path)

    # Check if the GIF has multiple frames
    if gif.is_animated:
        frames = []
        for frame in range(gif.n_frames):
            gif.seek(frame)
            # Reduce the size of each frame
            frame_image = gif.resize((gif.width // 2, gif.height // 2), Image.LANCZOS)
            frames.append(frame_image.convert("P", palette=Image.ADAPTIVE))

        # Save the GIF with reduced size and optimization
        frames[0].save(output_path, save_all=True, append_images=frames[1:], optimize=True, loop=0, quality=quality)
    else:
        # Reduce the size for a single-frame GIF
        gif = gif.resize((gif.width // 2, gif.height // 2), Image.LANCZOS)
        gif.save(output_path, optimize=True, quality=quality)

# Example usage

compress_gif('Minecraft_update_Google Chrome 2024-08-24 10-44-35.gif', 'output.gif', quality=50)

# %%



#%%

from moviepy.editor import *

# Load the image and audio
image = ImageClip("image128.png")
# image = ImageClip("image306.png")
image_clip = image
current_aspect_ratio = image_clip.w / image_clip.h
target_aspect_ratio = 16/9
if current_aspect_ratio > target_aspect_ratio:
    # Image is too wide, adjust the width
    new_width = int(image_clip.h * target_aspect_ratio)
    resized_clip = image_clip.resize(width=new_width)
else:
    # Image is too tall, adjust the height
    new_height = int(image_clip.w / target_aspect_ratio)
    resized_clip = image_clip.resize(height=new_height)

audio = AudioFileClip("20180127_taper2_SR_20k.wav")

# Set the duration of the video to match the audio's duration
video = image_clip.set_duration(audio.duration)

# Set the audio to the video
video = video.set_audio(audio)

# Set the frame rate (optional, default is 24)
video = video.set_fps(24)

# Write the video to an MP4 file
video.write_videofile("output_video.mp4", fps=24, codec="libx264", audio_codec="aac")



#%%
from moviepy.editor import ImageClip, AudioFileClip
from PIL import Image
import numpy as np

# Load the image and resize it to fit within 720p while maintaining aspect ratio
img = Image.open("image128.png")
img.thumbnail((1280, 720), Image.LANCZOS)

# Create a white background of 720p
background = Image.new('RGB', (1280, 720), color='white')

# Calculate position to paste the image
paste_position = ((1280 - img.width) // 2, (720 - img.height) // 2)

# Paste the image onto the white background
background.paste(img, paste_position)

# Convert PIL Image to numpy array for MoviePy
img_array = np.array(background)

# Create video clip from the image
video = ImageClip(img_array)

# Load audio file
audio = AudioFileClip("20180127_taper2_SR_20k.wav")

# Set duration of video to match audio
video = video.set_duration(audio.duration)

# Add audio to video
final_video = video.set_audio(audio)

# Write the result to a file
final_video.write_videofile('output_video.mp4', fps=24, codec='libx264', audio_codec="aac")

print("Video created: output_video.mp4")
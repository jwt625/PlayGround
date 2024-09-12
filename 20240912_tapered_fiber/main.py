

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






#%%


# Input and output file paths
input_file = 'EC_no_tether_20200121_addedMonitor_Si_200nm_monitor_4.mpg'
output_file = 'EC_no_tether_20200121_addedMonitor_Si_200nm_monitor_4.mp4'




import cv2

# Input and output file paths
# input_file = 'input_video.mpg'
# output_file = 'output_video.mp4'

# Open the mpg video file
cap = cv2.VideoCapture(input_file)

# Get the frames per second (fps) and the resolution of the input video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec for the output video (using MP4V codec)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter object to write the video to mp4
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Read and write frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

# Release everything once done
cap.release()
out.release()
cv2.destroyAllWindows()




#%% further cleanup the format
from moviepy.editor import VideoFileClip

# Input and output file paths
input_file = 'EC_no_tether_20200121_addedMonitor_Si_200nm_monitor_4.mp4'
output_file = 'EC_no_tether_20200121_addedMonitor_Si_200nm_monitor_4_clean.mp4'

# Load the mpg video
video_clip = VideoFileClip(input_file)

# Write the video to mp4 format with libx264 and aac for audio
video_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")


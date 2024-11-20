
#%%
import argparse
from moviepy.editor import VideoFileClip


#%%
def crop_video(input_file, output_file, top, bottom, left, right, trim_end=1):
    # Load the video clip
    clip = VideoFileClip(input_file)
    
    # Get the original dimensions
    width, height = clip.w, clip.h
    original_duration = clip.duration
    
    # Calculate the new dimensions
    new_width = width - left - right
    new_height = height - top - bottom
    
    # Crop the video
    cropped_clip = clip.crop(x1=left, y1=top, x2=width-right, y2=height-bottom)
    # Trim the last second
    new_duration = max(0, original_duration - trim_end)
    cropped_clip = cropped_clip.subclip(0, new_duration)
    
    # Export the video with Twitter-friendly settings
    cropped_clip.write_videofile(
        output_file,
        codec='libx264',
        audio_codec='aac',
        # temp_audiofile='temp-audio.m4a',
        temp_audiofile='',
        remove_temp=True,
        bitrate='8000k',
        fps=clip.fps
    )
    
    # Close the clips
    clip.close()
    cropped_clip.close()


input_file="input.mp4"
output_file="output.mp4"
top = 150
bottom = 124
left = 234
right = 250

crop_video(input_file, output_file, top, bottom, left, right)
print(f"Video cropped and exported successfully: {output_file}")



# %%
import argparse
from moviepy.editor import VideoFileClip

def crop_video_duration(input_file, output_file, start_time=0, end_time=None):
    """
    Crop the duration of a video file.
    
    :param input_file: Path to the input video file
    :param output_file: Path to save the output video file
    :param start_time: Time to start the video (in seconds, default 0)
    :param end_time: Time to end the video (in seconds, default None which means end of video)
    """
    # Load the video clip
    clip = VideoFileClip(input_file)
    
    # Crop the video duration
    cropped_clip = clip.subclip(start_time, end_time)
    
    # Export the video with original settings
    cropped_clip.write_videofile(
        output_file,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True,
        fps=30
    )
    
    # Close the clips
    clip.close()
    cropped_clip.close()

    
input_file="mmc5.mp4"
output_file="out.mp4"
crop_video_duration(input_file, output_file, start_time = 149.5, end_time = None)
print(f"Video duration cropped successfully: {output_file}")
# %%

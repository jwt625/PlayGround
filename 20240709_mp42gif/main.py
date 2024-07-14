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

#%% Example usage
mp4_path = "Animation_Different Quantities Animation.mp4"
gif_path = "Animation_Different Quantities Animation.gif"
convert_mp4_to_gif(mp4_path, gif_path)

# %%


# %% merge
import os
from moviepy.editor import VideoFileClip, clips_array

#%%
# def merge_mp4_videos(input_folder, output_file):

input_folder = "downloaded_mp4"  # Replace with the folder containing your MP4 videos
output_file = "merged_video.mp4"  # Replace with the desired output file name

video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

# if not video_files:
#     print("No MP4 files found in the input folder.")

video_clips = [VideoFileClip(os.path.join(input_folder, f)) for f in video_files]

# Concatenate the video clips
final_video = clips_array([[clip] for clip in video_clips])

# Write the final video to the output file
final_video.write_videofile(output_file, codec="libx264")
# final_video.write_videofile(output_file)

# merge_mp4_videos(input_folder, output_file)

# %%
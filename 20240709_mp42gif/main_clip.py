
#%%

from moviepy.video.io.VideoFileClip import VideoFileClip

def clip_video_for_twitter(input_path, output_path, duration=140):
    # Load the video
    video = VideoFileClip(input_path)
    
    # Clip the video to the first 'duration' seconds
    if video.duration > duration:
        clipped_video = video.subclip(0, duration)
    else:
        clipped_video = video  # If the video is shorter, no need to clip.
    
    # Resize the video if necessary to fit within Twitter's resolution
    # clipped_video = clipped_video.resize(height=1200)  # Resize to fit Twitter's max height
    
    # Save the output with correct codec and fps
    clipped_video.write_videofile(
        output_path, 
        codec="libx264",    # Video codec
        audio_codec="aac",  # Audio codec
        fps=30              # Frame rate
    )
    print(f"Video saved to {output_path}")

# Example usage
input_video = "highspeed_robot_hand.mp4"
output_video = "highspeed_robot_hand_clipped_video.mp4"
clip_video_for_twitter(input_video, output_video)

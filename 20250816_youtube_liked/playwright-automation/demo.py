"""
Demo script for testing YouTube video removal with a small number of videos.
This is useful for testing the automation before running large batches.
"""

from youtube_remover import YouTubeVideoRemover
from utils.logging import setup_logger

def main():
    """
    Demo the video removal process with a small number of videos.
    """
    logger = setup_logger()
    
    print("=== YouTube Video Removal Demo ===")
    print("This demo will attempt to remove 3 videos from your liked list.")
    print("Make sure you have a recent backup before proceeding!")
    print()
    print("Note: You'll be prompted to log in to YouTube if needed.")
    print("The login session will be saved for future use.")
    print()

    # Ask for confirmation
    response = input("Do you want to proceed? (y/N): ").strip().lower()
    if response != 'y':
        print("Demo cancelled.")
        return
    
    # Create remover instance (non-headless for demo)
    remover = YouTubeVideoRemover(headless=False)
    
    # Run the integrated workflow with just 3 videos
    logger.info("Starting demo with 3 video removals")
    success = remover.integrated_removal_workflow(videos_to_remove=3)
    
    if success:
        print("\nDemo completed successfully!")
        print("The automation is working correctly.")
        print("You can now run larger batches with confidence.")
    else:
        print("\nDemo encountered issues.")
        print("Check the logs for details and troubleshoot before running larger batches.")

if __name__ == "__main__":
    main()

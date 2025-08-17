"""
Demo script for testing YouTube video removal with a small number of videos.
This is useful for testing the automation before running large batches.
"""

import argparse
from youtube_remover import YouTubeVideoRemover
from utils.logging import setup_logger

def main():
    """
    Demo the video removal process with a small number of videos.
    """
    parser = argparse.ArgumentParser(description="Demo YouTube video removal")
    parser.add_argument("--count", type=int, default=3,
                       help="Number of videos to remove (default: 3)")
    args = parser.parse_args()

    logger = setup_logger()

    print("=== YouTube Video Removal Demo ===")
    print(f"This demo will attempt to remove {args.count} videos from your liked list.")
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

    # Run the integrated workflow with specified number of videos
    logger.info(f"Starting demo with {args.count} video removals")
    success = remover.integrated_removal_workflow(videos_to_remove=args.count)
    
    if success:
        print(f"\nDemo completed successfully!")
        print(f"Successfully removed {args.count} videos.")
        print("The automation is working correctly.")
        print("You can now run larger batches with confidence.")
    else:
        print(f"\nDemo encountered issues.")
        print("Check the logs for details and troubleshoot before running larger batches.")

if __name__ == "__main__":
    main()

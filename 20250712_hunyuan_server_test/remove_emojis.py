#!/usr/bin/env python3
"""
Simple script to remove all emojis from a text file.

Usage:
    python remove_emojis.py <filename>
    # or with uv:
    uv run remove_emojis.py <filename>

The script will read the file, remove all emojis, and overwrite the original file.
"""

import sys
import re
from pathlib import Path


def remove_emojis(text):
    """
    Remove all emojis from the given text using a simple but effective approach.

    This function removes characters from major emoji Unicode ranges.
    """
    # Simple but comprehensive emoji removal pattern
    emoji_pattern = re.compile(
        r"["
        r"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        r"\U0001F170-\U0001F251"  # enclosed alphanumeric supplement (includes 🆔)
        r"\U0001F300-\U0001F5FF"  # symbols & pictographs
        r"\U0001F600-\U0001F64F"  # emoticons
        r"\U0001F680-\U0001F6FF"  # transport & map symbols
        r"\U0001F700-\U0001F77F"  # alchemical symbols
        r"\U0001F780-\U0001F7FF"  # geometric shapes extended
        r"\U0001F800-\U0001F8FF"  # supplemental arrows-c
        r"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        r"\U0001FA00-\U0001FA6F"  # chess symbols
        r"\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        r"\U00002600-\U000026FF"  # miscellaneous symbols
        r"\U00002700-\U000027BF"  # dingbats
        r"\U0000FE00-\U0000FE0F"  # variation selectors
        r"\U00002190-\U000021FF"  # arrows
        r"\U00002B00-\U00002BFF"  # miscellaneous symbols and arrows
        r"\U0000231A-\U0000231B"  # watch
        r"\U000023E9-\U000023F3"  # various symbols
        r"\U000025AA-\U000025AB"  # squares
        r"\U000025B6\U000025C0"   # triangles
        r"\U000025FB-\U000025FE"  # squares
        r"\U00002B50\U00002B55"   # star and circle
        r"]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def process_file(filename):
    """
    Read a file, remove emojis, and write back to the same file.
    """
    file_path = Path(filename)
    
    # Check if file exists
    if not file_path.exists():
        print(f"Error: File '{filename}' not found.")
        return False
    
    # Check if it's a file (not a directory)
    if not file_path.is_file():
        print(f"Error: '{filename}' is not a file.")
        return False
    
    try:
        # Read the file
        print(f"Reading file: {filename}")
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Remove emojis
        print("Removing emojis...")
        cleaned_content = remove_emojis(original_content)
        
        # Count removed characters
        removed_count = len(original_content) - len(cleaned_content)
        
        # Write back to the same file
        print(f"Writing cleaned content back to: {filename}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"Success! Removed {removed_count} emoji characters.")
        return True
        
    except UnicodeDecodeError:
        print(f"Error: Could not decode '{filename}' as UTF-8 text.")
        return False
    except PermissionError:
        print(f"Error: Permission denied when accessing '{filename}'.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python remove_emojis.py <filename>")
        print("       uv run remove_emojis.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    success = process_file(filename)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

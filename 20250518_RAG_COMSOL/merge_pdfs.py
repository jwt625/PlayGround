import os
import shutil
from pathlib import Path

def merge_pdf_files():
    # Get the absolute path of the pdf folder
    pdf_folder = Path('pdf').absolute()
    
    # Create a set to track used filenames to handle duplicates
    used_filenames = set()
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(pdf_folder):
        root_path = Path(root)
        
        # Skip the main pdf folder itself
        if root_path == pdf_folder:
            continue
            
        # Process each file in the current directory
        for file in files:
            if file.lower().endswith('.pdf'):
                source_path = root_path / file
                
                # Create the destination filename
                dest_filename = file
                base_name = Path(file).stem
                extension = Path(file).suffix
                counter = 1
                
                # Handle duplicate filenames
                while dest_filename in used_filenames:
                    dest_filename = f"{base_name}_{counter}{extension}"
                    counter += 1
                
                dest_path = pdf_folder / dest_filename
                used_filenames.add(dest_filename)
                
                # Move the file
                try:
                    shutil.move(str(source_path), str(dest_path))
                    print(f"Moved: {source_path} -> {dest_path}")
                except Exception as e:
                    print(f"Error moving {source_path}: {str(e)}")

if __name__ == "__main__":
    print("Starting PDF file merge process...")
    merge_pdf_files()
    print("PDF file merge completed!") 
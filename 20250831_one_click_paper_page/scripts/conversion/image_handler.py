"""Image handling utilities for paper processing."""

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ImageHandler:
    """Utility class for handling image extraction and path processing."""

    def extract_and_save_images(self, result: Any, output_dir: Path) -> int:
        """Extract images from conversion result and save to images/ directory."""
        if not hasattr(result, 'images') or not result.images:
            logger.info("No images found in conversion result")
            return 0

        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        processed_count = 0
        for img_name, img_data in result.images.items():
            img_path = images_dir / img_name
            try:
                # Handle different image data types
                if hasattr(img_data, 'save'):
                    # PIL Image object
                    img_data.save(img_path)
                elif isinstance(img_data, bytes):
                    # Raw bytes
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                else:
                    logger.warning(
                        f"Unknown image data type for {img_name}: {type(img_data)}"
                    )
                    continue

                processed_count += 1
                logger.debug(f"Saved image: {img_path}")

            except Exception as e:
                logger.warning(f"Failed to save image {img_name}: {e}")

        logger.info(f"Processed {processed_count} images to {images_dir}")
        return processed_count

    def update_image_paths_in_markdown(self, markdown_content: str) -> str:
        """Update image paths in markdown to point to images/ directory."""
        # Pattern to match markdown images: ![alt text](image_path)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

        def update_path(match: re.Match[str]) -> str:
            alt_text = match.group(1)
            image_path = match.group(2)

            # Update the image path to point to images/ directory
            # Handle various path formats from marker
            if image_path.startswith('_page_') or not image_path.startswith('images/'):
                # If it's a marker-generated image name or doesn't start with images/
                updated_path = f"images/{image_path}"
            else:
                # Already has correct path
                updated_path = image_path

            return f'![{alt_text}]({updated_path})'

        # Replace all image references in the markdown
        updated_content = re.sub(image_pattern, update_path, markdown_content)

        # Log changes if any were made
        original_matches = re.findall(image_pattern, markdown_content)
        if original_matches:
            logger.info(f"Updated {len(original_matches)} image path(s) in markdown")

        return updated_content

    def process_images_in_html_line(self, line: str) -> str:
        """Convert markdown image syntax to HTML img tags with correct paths."""
        # Pattern to match markdown images: ![alt text](image_path)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

        def replace_image(match: re.Match[str]) -> str:
            alt_text = match.group(1)
            image_path = match.group(2)

            # Update the image path to point to images/ directory
            # Handle various path formats from marker
            if image_path.startswith('_page_') or not image_path.startswith('images/'):
                # If it's a marker-generated image name or doesn't start with images/
                updated_path = f"images/{image_path}"
            else:
                # Already has correct path
                updated_path = image_path

            # Create HTML img tag with responsive styling
            return (
                f'<img src="{updated_path}" alt="{alt_text}" '
                f'style="max-width: 100%; height: auto;">'
            )

        # Replace all image references in the line
        processed_line = re.sub(image_pattern, replace_image, line)

        return processed_line

    def count_images_in_content(self, content: str) -> int:
        """Count the number of images in HTML content."""
        return content.count('<img')

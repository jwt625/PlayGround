#!/usr/bin/env python3
"""
Apply theme template to the converted HTML content.
"""

import os
import sys
from pathlib import Path
import json
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default theme templates
ACADEMIC_THEME = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 2em;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .abstract {
            background-color: #f8f9fa;
            padding: 20px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }
        .authors {
            font-style: italic;
            margin-bottom: 20px;
            text-align: center;
        }
        .date {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
        }
        .math {
            text-align: center;
            margin: 20px 0;
        }
        .figure {
            text-align: center;
            margin: 20px 0;
        }
        .figure img {
            max-width: 100%;
            height: auto;
        }
        .figure-caption {
            font-style: italic;
            margin-top: 10px;
            color: #666;
        }
        .references {
            margin-top: 40px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
        .references h2 {
            color: #2c3e50;
        }
        .reference {
            margin-bottom: 10px;
            padding-left: 20px;
            text-indent: -20px;
        }
    </style>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }
        };
    </script>
</head>
<body>
    <header>
        <h1>{title}</h1>
        {authors_html}
        {date_html}
    </header>
    
    <main>
        {content}
    </main>
    
    <footer>
        <p style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
            Generated with <a href="https://github.com/your-username/paper-to-website">Paper to Website</a>
        </p>
    </footer>
</body>
</html>
"""

def load_config() -> Dict[str, Any]:
    """Load configuration from paper-config.json if it exists."""
    config_path = Path("paper-config.json")
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    return {
        "title": "Academic Paper",
        "authors": [],
        "theme": "academic",
        "date": ""
    }

def extract_metadata_from_html(html_content: str) -> Dict[str, Any]:
    """Extract metadata from HTML content."""
    metadata = {}
    
    # Simple extraction - in a real implementation, you'd use proper HTML parsing
    lines = html_content.split('\n')
    for line in lines:
        if '<title>' in line and '</title>' in line:
            title = line.split('<title>')[1].split('</title>')[0].strip()
            if title and title != "":
                metadata['title'] = title
                break
    
    return metadata

def apply_academic_theme(content: str, config: Dict[str, Any]) -> str:
    """Apply the academic theme to the content."""
    
    # Prepare authors HTML
    authors_html = ""
    if config.get('authors'):
        authors_list = config['authors']
        if isinstance(authors_list, list):
            authors_html = f'<div class="authors">{", ".join(authors_list)}</div>'
        else:
            authors_html = f'<div class="authors">{authors_list}</div>'
    
    # Prepare date HTML
    date_html = ""
    if config.get('date'):
        date_html = f'<div class="date">{config["date"]}</div>'
    
    # Apply template
    return ACADEMIC_THEME.format(
        title=config.get('title', 'Academic Paper'),
        authors_html=authors_html,
        date_html=date_html,
        content=content
    )

def main():
    """Main theme application logic."""
    dist_dir = Path("dist")
    index_file = dist_dir / "index.html"
    
    if not index_file.exists():
        logger.error("No index.html found in dist directory")
        sys.exit(1)
    
    # Load configuration
    config = load_config()
    
    # Read the converted HTML
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        logger.error(f"Failed to read index.html: {e}")
        sys.exit(1)
    
    # Extract metadata from HTML if not in config
    html_metadata = extract_metadata_from_html(html_content)
    for key, value in html_metadata.items():
        if key not in config or not config[key]:
            config[key] = value
    
    # Apply theme based on configuration
    theme = config.get('theme', 'academic')
    
    if theme == 'academic':
        themed_content = apply_academic_theme(html_content, config)
    else:
        logger.warning(f"Unknown theme '{theme}', using academic theme")
        themed_content = apply_academic_theme(html_content, config)
    
    # Write the themed content back
    try:
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(themed_content)
        logger.info("Theme applied successfully")
    except Exception as e:
        logger.error(f"Failed to write themed content: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

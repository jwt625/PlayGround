"""Metadata extraction utilities for paper processing."""

import logging
import re
from typing import Any

# Import text cleaning utilities
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils.text_cleaning import TextCleaner

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Utility class for extracting paper metadata from markdown content."""

    def extract_paper_metadata(self, markdown_content: str) -> dict[str, Any]:
        """
        Extract comprehensive paper metadata from markdown.

        Args:
            markdown_content: The markdown content to extract metadata from

        Returns:
            Dictionary containing extracted metadata
        """
        metadata: dict[str, Any] = {
            'title': None,
            'authors': [],
            'abstract': None,
            'keywords': [],
            'doi': None,
            'arxiv_id': None
        }

        try:
            lines = markdown_content.split('\n')

            # Extract title - look for headings and determine which is the actual
            # paper title
            headings = []
            for i, line in enumerate(lines[:30]):  # Check first 30 lines
                line = line.strip()
                if line.startswith('# ') and len(line) > 3:
                    heading_text = line[2:].strip()
                    headings.append((i, heading_text))

            # Determine which heading is the paper title
            paper_title = None

            for i, (line_num, heading_text) in enumerate(headings):
                # Check if this heading looks like arXiv metadata (first heading
                # pattern)
                if self.is_arxiv_metadata_heading(heading_text):
                    # Extract arXiv ID for metadata
                    arxiv_match = re.search(
                        r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)', heading_text, re.IGNORECASE
                    )
                    if arxiv_match:
                        metadata['arxiv_id'] = arxiv_match.group(1)
                    continue

                # Check if this heading looks like a paper title
                if self.is_paper_title_heading(heading_text):
                    paper_title = heading_text
                    break

            # If we found a paper title, clean it up
            if paper_title:
                cleaned_title = TextCleaner.clean_paper_title(paper_title)
                if cleaned_title and len(cleaned_title) > 3:
                    metadata['title'] = cleaned_title

            # Extract authors (look for patterns like "Author Name1, Author Name2")
            author_patterns = [
                # "John Doe, Jane Smith"
                r'^([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*)',
                r'^\*\*Authors?\*\*:?\s*(.+)',  # "**Authors**: John Doe, Jane Smith"
                r'^Authors?\s*:?\s*(.+)',       # "Authors: John Doe, Jane Smith"
            ]

            for line in lines[:50]:  # Check first 50 lines
                line = line.strip()
                for pattern in author_patterns:
                    match = re.match(pattern, line)
                    if match:
                        authors_text = match.group(1).strip()
                        # Split by common separators
                        authors = re.split(r',|;|\sand\s', authors_text)
                        authors = [
                            author.strip() for author in authors if author.strip()
                        ]
                        # Filter out non-author text
                        valid_authors = []
                        for author in authors:
                            # Simple heuristic: should contain at least first and
                            # last name
                            if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', author):
                                valid_authors.append(author)
                        if valid_authors:
                            metadata['authors'] = valid_authors
                            break
                if metadata['authors']:
                    break

            # Extract abstract
            abstract_start = -1
            for i, line in enumerate(lines):
                if re.match(r'^#+\s*abstract', line.strip(), re.IGNORECASE):
                    abstract_start = i + 1
                    break

            if abstract_start > 0:
                abstract_lines = []
                for i in range(abstract_start, min(abstract_start + 20, len(lines))):
                    line = lines[i].strip()
                    if line.startswith('#') or line.startswith('##'):
                        break
                    if line:
                        abstract_lines.append(line)

                if abstract_lines:
                    metadata['abstract'] = ' '.join(abstract_lines)

            # Extract DOI
            doi_pattern = r'(?:doi:|DOI:)?\s*(10\.\d+/[^\s]+)'
            for line in lines:
                match = re.search(doi_pattern, line, re.IGNORECASE)
                if match:
                    metadata['doi'] = match.group(1)
                    break

            # Extract arXiv ID
            arxiv_pattern = r'(?:arxiv:|arXiv:)?\s*(\d{4}\.\d{4,5}(?:v\d+)?)'
            for line in lines:
                match = re.search(arxiv_pattern, line, re.IGNORECASE)
                if match:
                    metadata['arxiv_id'] = match.group(1)
                    break

            logger.info(f"Extracted paper metadata: {metadata}")
            return metadata

        except Exception as e:
            logger.error(f"Paper metadata extraction error: {e}")
            return metadata



    def is_arxiv_metadata_heading(self, heading_text: str) -> bool:
        """
        Check if heading contains arXiv metadata rather than paper title.

        Args:
            heading_text: The heading text to check

        Returns:
            True if this looks like arXiv metadata, False otherwise
        """
        # Check for arXiv patterns
        arxiv_patterns = [
            r'arXiv:\d+\.\d+',  # arXiv:2506.15633
            r'\[arXiv:',        # [arXiv:...
            r'https?://arxiv',  # URLs to arXiv
        ]

        for pattern in arxiv_patterns:
            if re.search(pattern, heading_text, re.IGNORECASE):
                return True

        # Check for date patterns that suggest metadata
        date_patterns = [
            r'\d{1,2}\s+\w{3,9}\s+\d{4}',  # "18 Jun 2025"
            r'\d{4}-\d{2}-\d{2}',          # "2025-06-18"
        ]

        for pattern in date_patterns:
            if re.search(pattern, heading_text):
                return True

        # Check for subject classifications like [quant-ph], [cs.AI]
        if re.search(r'\[[a-z-]+\.[a-z-]+\]|\[[a-z-]+\]', heading_text, re.IGNORECASE):
            return True

        return False

    def is_paper_title_heading(self, heading_text: str) -> bool:
        """
        Check if heading looks like an actual paper title.

        Args:
            heading_text: The heading text to check

        Returns:
            True if this looks like a paper title, False otherwise
        """
        # Skip if it looks like arXiv metadata
        if self.is_arxiv_metadata_heading(heading_text):
            return False

        # Skip common non-title headings
        non_title_patterns = [
            r'^abstract$',
            r'^introduction$',
            r'^i\.\s*introduction$',
            r'^background$',
            r'^related\s+work$',
            r'^methodology$',
            r'^methods$',
            r'^results$',
            r'^conclusion$',
            r'^references$',
            r'^acknowledgments?$',
        ]

        for pattern in non_title_patterns:
            if re.match(pattern, heading_text.strip(), re.IGNORECASE):
                return False

        # A good paper title should:
        # 1. Be reasonably long (more than just a few words)
        # 2. Not be all uppercase (unless it's an acronym)
        # 3. Contain meaningful words

        # Clean the heading first
        clean_heading = TextCleaner.clean_paper_title(heading_text)

        if not clean_heading or len(clean_heading) < 10:
            return False

        # Check if it has reasonable content (not just numbers/symbols)
        word_count = len(re.findall(r'\b[a-zA-Z]+\b', clean_heading))
        if word_count < 3:
            return False

        return True



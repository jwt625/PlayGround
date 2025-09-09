"""Text cleaning utilities for paper title processing."""

import re


class TextCleaner:
    """Utility class for cleaning paper titles."""

    @staticmethod
    def clean_paper_title(title: str) -> str:
        """Clean paper title extracted from markdown."""
        if not title:
            return ""

        # Remove markdown formatting first
        title = re.sub(r'\*\*([^*]+)\*\*', r'\1', title)  # Remove bold
        title = re.sub(r'\*([^*]+)\*', r'\1', title)      # Remove italic
        title = re.sub(r'`([^`]+)`', r'\1', title)        # Remove code formatting

        # Remove markdown links: [text](url) -> text
        title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', title)

        # Remove URLs (both http and https)
        title = re.sub(r'https?://[^\s]+', '', title, flags=re.IGNORECASE)

        # Remove arXiv references (comprehensive patterns)
        title = re.sub(r'arXiv:\d+\.\d+v?\d*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\[arXiv:[^\]]+\]', '', title, flags=re.IGNORECASE)
        # [quant-ph], [cs.AI], etc.
        title = re.sub(r'\[[a-z.-]+\]', '', title, flags=re.IGNORECASE)

        # Remove date patterns
        # "18 Jun 2025", "1 January 2024"
        title = re.sub(r'\d{1,2}\s+\w{3,9}\s+\d{4}', '', title)
        title = re.sub(r'\d{4}-\d{2}-\d{2}', '', title)          # "2025-06-18"
        title = re.sub(r'\d{2}/\d{2}/\d{4}', '', title)          # "06/18/2025"

        # Remove version indicators
        title = re.sub(r'v\d+', '', title, flags=re.IGNORECASE)

        # Remove common prefixes/suffixes
        title = re.sub(
            r'^(paper|document|draft|manuscript):\s*', '', title, flags=re.IGNORECASE
        )
        title = re.sub(
            r'\s*(paper|document|draft|manuscript)$', '', title, flags=re.IGNORECASE
        )

        # Remove extra whitespace and normalize
        title = re.sub(r'\s+', ' ', title).strip()

        # Remove leading/trailing punctuation
        title = re.sub(r'^[^\w]+|[^\w]+$', '', title)

        return title

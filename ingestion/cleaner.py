# ingestion/cleaner.py
"""
Cleaner module: extracts meaningful text from raw HTML.
"""

from bs4 import BeautifulSoup
import re

UNWANTED_SELECTORS = [
    "nav",
    "footer",
    "header",
    "aside",
    "script",
    "style",
    "[role='navigation']",
    "[aria-label*='cookie']",
    ".cookie",
    ".cookies",
    ".cookie-banner",
    ".cookie-consent",
]


def clean_html(html_content: str) -> str:
    """Extract clean text from HTML."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove unwanted tags
    for tag in soup(UNWANTED_SELECTORS):
        tag.decompose()
    
    # Remove "edit" links (Wikipedia specific)
    for span in soup.find_all("span", class_="mw-editsection"):
        span.decompose()
    
    # Extract text from paragraphs
    text_blocks = []
    for paragraph in soup.find_all("p"):
        para_text = paragraph.get_text(strip=True)
        if para_text:
            text_blocks.append(para_text)
    
    # Combine and clean up extra newlines
    cleaned_text = "\n".join(text_blocks)
    cleaned_text = re.sub(r"\n{2,}", "\n", cleaned_text)
    
    return cleaned_text

# ingestion/crawler.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import List, Dict, Optional
import logging

# Get logger
logger = logging.getLogger("rag.crawler")

# Configuration
MAX_DEPTH = 2
MAX_PAGES = 20
REQUEST_TIMEOUT = 15
MIN_CONTENT_LENGTH = 100  # Minimum characters for valid content

SKIP_KEYWORDS = ["login", "signup", "register", "cart", "checkout", "privacy", "terms", "cookie"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9"
}


class CrawlError(Exception):
    """Custom exception for crawl errors."""
    pass


class NetworkError(CrawlError):
    """Network-related crawl error."""
    pass


class ContentError(CrawlError):
    """Content-related crawl error."""
    pass


def is_valid_internal_link(base_netloc: str, url: str) -> bool:
    """Check if URL is an internal link."""
    parsed = urlparse(url)
    return parsed.netloc == "" or parsed.netloc == base_netloc


def should_skip(url: str) -> bool:
    """Check if URL should be skipped based on keywords."""
    return any(keyword in url.lower() for keyword in SKIP_KEYWORDS)


def fetch_page(url: str) -> Optional[requests.Response]:
    """
    Fetch a single page with comprehensive error handling.
    
    Returns:
        Response object or None if fetch failed
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        
        # Check status code
        if response.status_code == 404:
            logger.debug(f"Page not found: {url}")
            return None
        elif response.status_code == 403:
            logger.warning(f"Access forbidden: {url}")
            return None
        elif response.status_code == 429:
            logger.warning(f"Rate limited: {url}")
            return None
        elif response.status_code >= 500:
            logger.warning(f"Server error ({response.status_code}): {url}")
            return None
        elif response.status_code != 200:
            logger.debug(f"Non-200 status ({response.status_code}): {url}")
            return None
        
        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            logger.debug(f"Non-HTML content type ({content_type}): {url}")
            return None
        
        return response
        
    except requests.exceptions.Timeout:
        logger.warning(f"Request timeout: {url}")
        return None
    except requests.exceptions.TooManyRedirects:
        logger.warning(f"Too many redirects: {url}")
        return None
    except requests.exceptions.SSLError as e:
        logger.warning(f"SSL error for {url}: {str(e)}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error for {url}: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {str(e)}")
        return None


def extract_page_content(response: requests.Response, url: str) -> Optional[Dict]:
    """
    Extract content from a page response.
    
    Returns:
        Dict with url, title, html or None if content is invalid
    """
    try:
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        
        # Check for meaningful content (not empty or error pages)
        text_content = soup.get_text(strip=True)
        if len(text_content) < MIN_CONTENT_LENGTH:
            logger.debug(f"Page has too little content ({len(text_content)} chars): {url}")
            return None
        
        return {
            "url": url,
            "title": title,
            "html": response.text
        }
        
    except Exception as e:
        logger.warning(f"Failed to parse page {url}: {str(e)}")
        return None


def crawl_site(base_url: str) -> List[Dict]:
    """
    Crawl a website starting from base_url.
    
    Args:
        base_url: Starting URL to crawl
        
    Returns:
        List of dicts with url, title, html
        
    Raises:
        NetworkError: If the base URL cannot be reached
        ContentError: If no valid content could be extracted
    """
    logger.info(f"Starting crawl of {base_url}")
    
    parsed_base = urlparse(base_url)
    base_netloc = parsed_base.netloc
    
    if not base_netloc:
        raise CrawlError(f"Invalid base URL: {base_url}")

    visited = set()
    results = []
    failed_urls = []
    
    queue = deque([(base_url, 0)])
    
    # First, verify the base URL is reachable
    initial_response = fetch_page(base_url)
    if initial_response is None:
        raise NetworkError(f"Cannot reach {base_url}. Please check the URL and your network connection.")

    while queue and len(results) < MAX_PAGES:
        current_url, depth = queue.popleft()

        if current_url in visited or depth > MAX_DEPTH:
            continue
        if should_skip(current_url):
            logger.debug(f"Skipping URL (keyword match): {current_url}")
            continue

        visited.add(current_url)
        
        # Use cached response for base URL
        if current_url == base_url and initial_response:
            response = initial_response
            initial_response = None  # Clear cache after use
        else:
            response = fetch_page(current_url)
        
        if response is None:
            failed_urls.append(current_url)
            continue
        
        # Extract content
        page_data = extract_page_content(response, current_url)
        if page_data:
            results.append(page_data)
            logger.debug(f"Crawled: {current_url} (title: {page_data['title'][:50]}...)")
        
        # Extract and queue links
        try:
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                absolute_url = urljoin(current_url, link["href"])
                absolute_url = absolute_url.split("#")[0].split("?")[0]  # Remove fragments and query params
                
                if absolute_url and is_valid_internal_link(base_netloc, absolute_url):
                    if absolute_url not in visited:
                        queue.append((absolute_url, depth + 1))
        except Exception as e:
            logger.warning(f"Failed to extract links from {current_url}: {str(e)}")

    # Log summary
    logger.info(f"Crawl complete: {len(results)} pages crawled, {len(failed_urls)} failed, {len(visited)} visited")
    
    if not results:
        if failed_urls:
            raise NetworkError(f"Crawl failed: Could not fetch any pages. {len(failed_urls)} URLs failed.")
        else:
            raise ContentError("No valid content found on the website.")
    
    return results

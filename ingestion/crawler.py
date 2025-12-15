# ingestion/crawler.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

MAX_DEPTH = 2
MAX_PAGES = 20

SKIP_KEYWORDS = ["login", "signup", "register", "cart", "checkout", "privacy", "terms"]

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html"
}


def is_valid_internal_link(base_netloc, url):
    parsed = urlparse(url)
    return parsed.netloc == "" or parsed.netloc == base_netloc


def should_skip(url):
    return any(keyword in url.lower() for keyword in SKIP_KEYWORDS)


def crawl_site(base_url):
    parsed_base = urlparse(base_url)
    base_netloc = parsed_base.netloc

    visited = set()
    results = []

    queue = deque([(base_url, 0)])

    while queue and len(results) < MAX_PAGES:
        current_url, depth = queue.popleft()

        if current_url in visited or depth > MAX_DEPTH:
            continue
        if should_skip(current_url):
            continue

        try:
            response = requests.get(current_url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                continue
            if "text/html" not in response.headers.get("Content-Type", ""):
                continue
        except requests.RequestException:
            continue

        visited.add(current_url)

        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title else ""

        results.append({
            "url": current_url,
            "title": title,
            "html": response.text
        })

        # Extract links
        for link in soup.find_all("a", href=True):
            absolute_url = urljoin(current_url, link["href"])
            absolute_url = absolute_url.split("#")[0]

            if is_valid_internal_link(base_netloc, absolute_url):
                if absolute_url not in visited:
                    queue.append((absolute_url, depth + 1))

    return results

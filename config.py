# config.py - Centralized configuration for RAG Chatbot
"""
All limits and configuration values in one place.

DEMO MODE: Optimized for smooth demonstrations.
For production, consider stricter rate limits and larger crawl timeouts.
"""

# === INPUT LIMITS ===
MAX_QUESTION_LENGTH = 2000          # Max characters for a question
MAX_URL_LENGTH = 2048               # Max characters for a URL
MAX_SESSION_ID_LENGTH = 64          # Max characters for session ID

# === FILE LIMITS ===
MAX_FILE_SIZE_MB = 15               # Max file upload size in MB (generous for demo)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_FILE_EXTENSIONS = {".txt", ".pdf", ".docx"}

# === CRAWL LIMITS ===
MAX_CRAWL_DEPTH = 1                 # Keep shallow for fast demo (1 = homepage + direct links only)
MAX_PAGES_PER_CRAWL = 10            # Fewer pages = faster crawl demo
CRAWL_TIMEOUT_SECONDS = 10          # Quick timeout to avoid hanging
MIN_PAGE_CONTENT_LENGTH = 50        # Accept smaller pages

# === RATE LIMITING (DEMO: Very generous) ===
RATE_LIMIT_REQUESTS = 100           # 100 requests per minute (won't block during demo)
RATE_LIMIT_WINDOW_SECONDS = 60      # Time window in seconds

# === RAG SETTINGS ===
MAX_CHAT_HISTORY_MESSAGES = 10      # Sliding window for conversation memory
RETRIEVER_TOP_K = 4                 # Retrieve more docs for better answers

# === CHUNK SETTINGS ===
CHUNK_SIZE = 500                    # Smaller chunks = more precise retrieval
CHUNK_OVERLAP = 100                 # Overlap between chunks

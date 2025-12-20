# utils/logger.py - Centralized logging configuration for RAG pipeline
import logging
import sys
from datetime import datetime

# Log format with timestamp, level, module, and message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger instance."""
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(console_handler)
    
    return logger


# Pre-configured loggers for different modules
def get_server_logger():
    """Logger for API server operations."""
    return setup_logger("rag.server")


def get_crawler_logger():
    """Logger for web crawling operations."""
    return setup_logger("rag.crawler")


def get_ingestion_logger():
    """Logger for document ingestion pipeline."""
    return setup_logger("rag.ingestion")


def get_rag_logger():
    """Logger for RAG chain operations."""
    return setup_logger("rag.chain")

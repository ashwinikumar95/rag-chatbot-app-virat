# utils/validators.py - Input validation utilities
import re
from urllib.parse import urlparse
from typing import Optional


class URLValidationError(Exception):
    """Custom exception for URL validation errors."""
    pass


def validate_url(url: str) -> str:
    """
    Validate and normalize a URL.
    
    Args:
        url: The URL string to validate
        
    Returns:
        Normalized URL string
        
    Raises:
        URLValidationError: If URL is invalid
    """
    if not url or not isinstance(url, str):
        raise URLValidationError("URL is required and must be a string")
    
    url = url.strip()
    
    if not url:
        raise URLValidationError("URL cannot be empty")
    
    # Check for minimum length
    if len(url) < 10:
        raise URLValidationError("URL is too short to be valid")
    
    # Check for maximum length (prevent DoS)
    if len(url) > 2048:
        raise URLValidationError("URL exceeds maximum allowed length (2048 characters)")
    
    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise URLValidationError(f"Failed to parse URL: {str(e)}")
    
    # Validate scheme
    if not parsed.scheme:
        raise URLValidationError("URL must include a scheme (http:// or https://)")
    
    if parsed.scheme.lower() not in ("http", "https"):
        raise URLValidationError(f"Invalid URL scheme '{parsed.scheme}'. Only http and https are supported")
    
    # Validate netloc (domain)
    if not parsed.netloc:
        raise URLValidationError("URL must include a valid domain")
    
    # Basic domain validation
    domain = parsed.netloc.lower()
    
    # Check for localhost/internal IPs (optional - could block for security)
    blocked_domains = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
    if any(domain.startswith(blocked) for blocked in blocked_domains):
        raise URLValidationError("Local/internal URLs are not allowed")
    
    # Check for valid domain format
    domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}(:\d+)?$'
    # Also allow IP addresses
    ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$'
    
    if not (re.match(domain_pattern, domain) or re.match(ip_pattern, domain)):
        raise URLValidationError(f"Invalid domain format: {domain}")
    
    return url


def validate_session_id(session_id: str) -> str:
    """
    Validate session ID format.
    
    Args:
        session_id: The session ID to validate
        
    Returns:
        Validated session ID
        
    Raises:
        ValueError: If session ID is invalid
    """
    if not session_id or not isinstance(session_id, str):
        raise ValueError("session_id is required")
    
    session_id = session_id.strip()
    
    if not session_id:
        raise ValueError("session_id cannot be empty")
    
    # Allow alphanumeric, underscore, hyphen
    if not re.match(r'^[a-zA-Z0-9_\-]+$', session_id):
        raise ValueError("session_id contains invalid characters. Use only alphanumeric, underscore, or hyphen")
    
    if len(session_id) > 64:
        raise ValueError("session_id exceeds maximum length (64 characters)")
    
    return session_id

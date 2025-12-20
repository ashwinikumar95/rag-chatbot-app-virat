# utils/rate_limiter.py - Simple in-memory rate limiter
"""
Simple token bucket rate limiter for API endpoints.
Production use: Replace with Redis-based rate limiter.
"""

import time
from collections import defaultdict
from fastapi import HTTPException, Request
from typing import Dict, Tuple

# In-memory storage for rate limiting
# Structure: {client_ip: (request_count, window_start_time)}
_rate_limit_store: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, 0.0))


def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    # Check for forwarded IP (behind proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fall back to direct client
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request, max_requests: int, window_seconds: int) -> None:
    """
    Check if request is within rate limits.
    
    Args:
        request: FastAPI request object
        max_requests: Maximum requests allowed per window
        window_seconds: Time window in seconds
        
    Raises:
        HTTPException: 429 Too Many Requests if limit exceeded
    """
    client_ip = get_client_ip(request)
    current_time = time.time()
    
    count, window_start = _rate_limit_store[client_ip]
    
    # Check if we're in a new window
    if current_time - window_start > window_seconds:
        # Reset the window
        _rate_limit_store[client_ip] = (1, current_time)
        return
    
    # Check if limit exceeded
    if count >= max_requests:
        retry_after = int(window_seconds - (current_time - window_start))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )
    
    # Increment counter
    _rate_limit_store[client_ip] = (count + 1, window_start)


def reset_rate_limit(client_ip: str) -> None:
    """Reset rate limit for a specific client (for testing)."""
    if client_ip in _rate_limit_store:
        del _rate_limit_store[client_ip]

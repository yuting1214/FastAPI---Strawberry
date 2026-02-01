"""
Initialize settings based on command-line arguments and environment.

Usage:
    # Import in main.py to get mode-aware settings
    from app.core.init_settings import settings

    # Or run directly
    python -m app.main --mode prod --host 0.0.0.0
"""
import os
import sys
import argparse
from app.core.config import get_settings

# Port from environment (Railway injects PORT) or default
DEFAULT_PORT = int(os.getenv("PORT", "8000"))

# Set up argument parser
parser = argparse.ArgumentParser(description="GraphQL API Server")
parser.add_argument(
    "--mode",
    choices=["dev", "prod"],
    default="dev",
    help="Running mode: dev (SQLite) or prod (PostgreSQL)"
)
parser.add_argument(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host to bind to"
)
parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_PORT,
    help="Port to bind to (default: $PORT or 8000)"
)

# Check if running under pytest or uvicorn reload
is_testing = "pytest" in sys.argv[0]
is_uvicorn = "uvicorn" in sys.argv[0]

if is_testing or is_uvicorn:
    # Use default dev settings when running tests or via uvicorn
    mode = os.getenv("APP_MODE", "dev")
    args = argparse.Namespace(mode=mode, host="127.0.0.1", port=DEFAULT_PORT)
else:
    # Parse arguments when running directly
    args = parser.parse_args()

# Initialize settings based on mode
settings = get_settings(args.mode)

# Export for use in other modules
__all__ = ["settings", "args"]

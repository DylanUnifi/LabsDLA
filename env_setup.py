"""
Environment setup utility for LabsDLA.

Loads API keys and tokens from a .env file at the project root,
so they are never hardcoded in notebooks or scripts.

Usage (first cell of every notebook):
    from env_setup import setup_env
    setup_env()
"""

import os
from pathlib import Path


def setup_env(dotenv_path: str | None = None) -> None:
    """Load environment variables from a .env file.

    Searches for .env in the current directory and parent directories
    (up to 3 levels) to work from any Lab subdirectory.

    Args:
        dotenv_path: Optional explicit path to a .env file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("python-dotenv not installed. Run: pip install python-dotenv")
        print("   Falling back to existing environment variables.")
        return

    if dotenv_path and os.path.isfile(dotenv_path):
        load_dotenv(dotenv_path, override=False)
        return

    # Search current dir and up to 3 parent directories for .env
    search_dir = Path.cwd()
    for _ in range(4):
        env_file = search_dir / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            _print_status()
            return
        search_dir = search_dir.parent

    print("No .env file found. Copy .env.example to .env and fill in your keys:")
    print("      cp .env.example .env")


def _print_status() -> None:
    """Print which keys are loaded (without revealing values)."""
    keys = ["WANDB_API_KEY", "HF_TOKEN"]
    loaded = [k for k in keys if os.environ.get(k)]
    missing = [k for k in keys if not os.environ.get(k)]

    if loaded:
        print(f"Loaded from .env: {', '.join(loaded)}")
    if missing:
        print(f"Missing in .env: {', '.join(missing)}")

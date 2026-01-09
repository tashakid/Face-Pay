"""
Debug mode configuration for face recognition system.
Controls verbosity of logging based on environment variable.
"""

import os
import logging

def configure_debug_mode():
    """
    Configure logging level based on DEBUG_MODE environment variable.

    Set DEBUG_MODE=true to enable detailed debug logging.
    Default is INFO level logging.
    """
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"

    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        print("🐛 DEBUG MODE ENABLED - Verbose logging active")
    else:
        logging.getLogger().setLevel(logging.INFO)
        print("ℹ️  Standard logging enabled")

    return debug_mode
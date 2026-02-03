import logging
import os

_CONFIGURED = False

def setup_logging(level = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True
    )

    _CONFIGURED = True

def get_logger(name):
    return logging.getLogger(name)
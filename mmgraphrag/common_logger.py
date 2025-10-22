import logging
import os


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("multimodal-graphrag")


def get_logger(name: str = None) -> logging.Logger:
    if name:
        return logging.getLogger(name)
    return logger

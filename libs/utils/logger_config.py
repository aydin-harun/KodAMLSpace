import logging
from logging.config import dictConfig
import os

def setup_logging(log_dir="logs", log_level="INFO"):
    os.makedirs(log_dir, exist_ok=True)

    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            },
            "json": {
                "format": '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": "%(message)s"}',
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": log_level,
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": os.path.join(log_dir, "app.log"),
                "when": "midnight",
                "backupCount": 7,
                "formatter": "default",
                "encoding": "utf8",
                "level": log_level,
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": log_level,
        },
    })

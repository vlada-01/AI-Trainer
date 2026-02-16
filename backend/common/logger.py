import logging
import logging.config
import os
import colorlog

CONFIGURED = False

def setup_logging():
    global CONFIGURED
    if CONFIGURED:
        return

    lvl_str = (os.getenv("LOG_LEVEL", "DEBUG") or "DEBUG").upper()
    lvl = getattr(logging, lvl_str, logging.INFO)
    mute_others = os.getenv("MUTE_THIRD_PARTY_LOGS", "1") == "1"
    root_lvl = logging.WARNING if mute_others else lvl
    
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": mute_others,
        "formatters": {
            "colored": {
                "()": "colorlog.ColoredFormatter",
                "format": "%(log_color)s%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                "log_colors": {
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "colored",
                "level": lvl
            }
        },
        "loggers": {
            "app_src": {"handlers": ["console"], "level": lvl, "propagate": False},
            "model_src": {"handlers": ["console"], "level": lvl, "propagate": False},
            },
        "root": {"handlers": ["console"], "level": root_lvl},
    })

    CONFIGURED = True

def get_logger(name):
    return logging.getLogger(name)
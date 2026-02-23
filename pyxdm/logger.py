"""Logging configuration for the PyXDM package."""

import logging
import logging.config
import os
from typing import Any

from . import __version__

logger = logging.getLogger(__name__)

try:
    import horton as ht

    class SelectiveScreenLog(ht.log.__class__):
        def __call__(self, *a, **k):
            if self._level >= self.warning:
                super().__call__(*a, **k)

        def warn(self, *a: Any, **k: Any) -> None:
            super().warn(*a, **k)

        def hline(self, *a: Any, **k: Any) -> None:
            pass

        def center(self, *a: Any, **k: Any) -> None:
            pass

        def blank(self, *a: Any, **k: Any) -> None:
            pass

        def deflist(self, *a: Any, **k: Any) -> None:
            pass

        def progress(self, *a: Any, **k: Any) -> None:
            if self._level == self.medium:
                return super().progress(*a, **k)
            return lambda *aa, **kk: None

        def print_header(self, *a: Any, **k: Any) -> None:
            pass

        def print_footer(self, *a: Any, **k: Any) -> None:
            pass

    ht.log.__class__ = SelectiveScreenLog
    ht.log.set_level(0)

except ImportError:
    logger.warning("Horton is not installed. Some features may not work as expected. Please install Horton to enable full functionality.")


def log_banner() -> None:
    """Print a banner with the PyXDM logo."""
    banner = r"""


+=====================================================+
|                                                     |
|                                                     |
|  __________        ____  ___________      _____     |
|  \______   \___.__.\   \/  /\______ \    /     \    |
|   |     ___<   |  | \     /  |    |  \  /  \ /  \   |
|   |    |    \___  | /     \  |    `   \/    Y    \  |
|   |____|    / ____|/___/\  \/_______  /\____|__  /  |
|             \/           \_/        \/         \/   |
|                                                     |
|                                                     |
|  version: {}                                     |
+=====================================================+


""".format(__version__)

    lines = banner.split("\n")

    # Iterate over each line and log them
    for line in lines[1:-1]:
        logger.info(line)


class BlockFilter(logging.Filter):
    """Filter to block loggers not starting with pyxdm."""

    def filter(self, record):
        """Filter loggers not starting with pyxdm."""
        return record.name.startswith("pyxdm")


def config_logger() -> None:
    """Configure the logger for the PyXDM package."""
    # Define log level
    log_level = os.environ.get("PYXDM_LOG_LEVEL", default="INFO").upper()
    silence_loggers = int(os.environ.get("PYXDM_FILTER_LOGGERS", default=1))

    fmt = "%(asctime)s %(levelname)-8s %(name)-30s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Set up basicConfig
    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": fmt, "datefmt": datefmt}},
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            }
        },
        "loggers": {
            "": {"handlers": ["stdout"], "level": log_level},
        },
    }

    logging.config.dictConfig(LOGGING)

    try:
        import coloredlogs

        coloredlogs.install(level=getattr(logging, log_level), fmt=fmt, datefmt=datefmt)
    except ImportError:
        pass

    if silence_loggers:
        for handler in logging.getLogger().handlers:
            handler.addFilter(BlockFilter())


config_logger()
log_banner()

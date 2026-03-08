"""Logging utilities"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import TimedRotatingFileHandler

import yaml

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def _get_level(level_str: str) -> int:
    return getattr(logging, level_str.upper(), logging.INFO)

def get_logger(level: Optional[str] = None) -> logging.Logger:
    """return module-level logger"""

    logger = logging.getLogger(__name__)

    if logger.handlers:
        return logger
    
    if level is None:
        try:
            cfg_path = Path(__file__).parents[2] / "config" / "base_config.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            level = cfg.get("logging", {}).get("level", "INFO")
        except Exception:
            level = "INFO"

    logger.setLevel(_get_level(level_str=level))

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_FORMATTER)
    logger.addHandler(sh)

    try:
        fh = TimedRotatingFileHandler(
            filename=_LOG_DIR / "pipeline.log",
            when="W0",
            backupCount=4,
            encoding="utf-8"
        )

        fh.setFormatter(_FORMATTER)
        logger.addHandler(fh)
    except Exception as exc:
        logger.warning("Could not attach file handler: %s", exc)
    
    logger.propagate = False

    return logger

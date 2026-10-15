"""
NovaCart AI Business Intelligence System - Shared Utilities
Provides path resolution, logging setup, metric serialization, and configuration constants.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
APP_DIR = PROJECT_ROOT / "app"

# Ensure runtime directories exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, APP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random seed for global reproducibility
RANDOM_STATE = 42

def setup_logger(name: str = "novacart", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standardized console logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Saves a dictionary to JSON format with indentation.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)

def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Loads JSON data from file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

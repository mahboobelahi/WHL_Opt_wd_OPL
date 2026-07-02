"""Repository-local path constants for the warehouse layout revision project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
INSTANCE_DIR = DATA_DIR / "instances"
MASK_DIR = INSTANCE_DIR / "masks"
REFERENCE_LAYOUT_DIR = INSTANCE_DIR / "reference_layouts"
RAW_RESULTS_DIR = DATA_DIR / "raw_results"
PROCESSED_RESULTS_DIR = DATA_DIR / "processed_results"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"

__all__ = [
    "CONFIG_DIR",
    "DATA_DIR",
    "DOCS_DIR",
    "FIGURES_DIR",
    "INSTANCE_DIR",
    "LOGS_DIR",
    "MASK_DIR",
    "PROCESSED_RESULTS_DIR",
    "PROJECT_ROOT",
    "RAW_RESULTS_DIR",
    "REFERENCE_LAYOUT_DIR",
    "RESULTS_DIR",
    "TABLES_DIR",
]

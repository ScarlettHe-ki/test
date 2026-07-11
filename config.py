"""Shared paths for the subgroup-scanning pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
CLEANED_REGION_DATA_PATH = DATA_DIR / "processed_data_region_clean.csv"
REPORTS_DIR = ROOT_DIR / "reports"

# Which file 01_sparsity_dispersion_check.py reads. Point this at
# CLEANED_REGION_DATA_PATH to run diagnostics on the cleaned + coarsened
# geography data produced by 00_clean_loss_region.py.
ACTIVE_DATA_PATH = PROCESSED_DATA_PATH

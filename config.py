"""Shared paths for the subgroup-scanning pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
CLEANED_REGION_DATA_PATH = DATA_DIR / "processed_data_region_clean.csv"
REPORTS_DIR = ROOT_DIR / "reports"
ANCHORED_EXPECTED_COUNTS_PATH = REPORTS_DIR / "expected_counts_anchored_test.csv"

# Which file the pipeline scripts (01, 04, ...) read by default. The frozen
# design (from 00-03) runs on the cleaned + coarsened geography data; point
# this back at PROCESSED_DATA_PATH only to re-check diagnostics on the raw file.
ACTIVE_DATA_PATH = CLEANED_REGION_DATA_PATH

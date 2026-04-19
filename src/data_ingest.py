"""
data_ingest.py
--------------
Downloads the UCI Default of Credit Card Clients dataset and saves a clean
CSV to data/staged/data.csv.

Source:
  Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques
  for the predictive accuracy of probability of default of credit card clients.
  Expert Systems with Applications, 36(2), 2473-2480.
  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
"""

import os
import logging
import requests
import zipfile
import io
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = os.path.join("data", "raw")
STAGED_DIR = os.path.join("data", "staged")
RAW_FILE = os.path.join(RAW_DIR, "credit_default_raw.xls")
STAGED_FILE = os.path.join(STAGED_DIR, "data.csv")

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)


def download_dataset():
    """Download the dataset zip from UCI and extract the XLS file."""
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(RAW_FILE):
        logger.info("Raw file already exists, skipping download.")
        return

    logger.info(f"Downloading dataset from {UCI_URL} ...")
    response = requests.get(UCI_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        xls_name = [n for n in zf.namelist() if n.endswith(".xls")][0]
        logger.info(f"Extracting {xls_name} ...")
        zf.extract(xls_name, RAW_DIR)
        extracted_path = os.path.join(RAW_DIR, xls_name)
        if extracted_path != RAW_FILE:
            os.rename(extracted_path, RAW_FILE)

    logger.info(f"Raw file saved to {RAW_FILE}")


def clean_and_stage():
    """Read raw XLS, normalise column names, and write staged CSV."""
    os.makedirs(STAGED_DIR, exist_ok=True)
    logger.info(f"Reading {RAW_FILE} ...")

    # Row 0 is metadata; actual header is row 1
    df = pd.read_excel(RAW_FILE, header=1, engine="xlrd")

    # Rename target column for clarity
    df.rename(columns={"default payment next month": "DEFAULT"}, inplace=True)

    # Drop the ID column – not a feature
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

    # Standardise column names to upper-case, no spaces
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Class distribution:\n{df['DEFAULT'].value_counts()}")

    df.to_csv(STAGED_FILE, index=False)
    logger.info(f"Staged data written to {STAGED_FILE}")


def main():
    download_dataset()
    clean_and_stage()
    logger.info("Data ingestion complete.")


if __name__ == "__main__":
    main()

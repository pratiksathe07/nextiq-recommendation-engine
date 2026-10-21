"""
NextIQ — Dataset Downloader and Sample Generator
Provides automated downloading of UCI Online Retail dataset,
with an offline realistic sample generation fallback.
"""

import argparse
import io
import logging
import os
from pathlib import Path
import random
import sys
import zipfile
import numpy as np
import pandas as pd
import requests

# Add repo root to python path for config imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import (
    COL_COUNTRY,
    COL_CUSTOMER_ID,
    COL_DESCRIPTION,
    COL_INVOICE_DATE,
    COL_INVOICE_NO,
    COL_QUANTITY,
    COL_STOCK_CODE,
    COL_UNIT_PRICE,
    DATA_DIR,
    RAW_CSV_PATH,
    RAW_DATA_PATH,
    SAMPLE_DATA_PATH,
)
from src.data_loader import validate_schema

logger = logging.getLogger("NextIQ.DataDownloader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
GITHUB_CSV_MIRROR = "https://raw.githubusercontent.com/datagy/data/main/online-retail-dataset.csv"


def download_uci_dataset(dest_dir: Path, force: bool = False) -> Path:
    """
    Downloads and extracts the official UCI Online Retail dataset.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_target = dest_dir / "online_retail.csv"
    xlsx_target = dest_dir / "online_retail.xlsx"

    if not force:
        if csv_target.exists():
            logger.info(f"Existing dataset found at {csv_target}. Skipping download.")
            return csv_target
        if xlsx_target.exists():
            logger.info(f"Existing dataset found at {xlsx_target}. Skipping download.")
            return xlsx_target

    logger.info(f"Attempting download from UCI repository: {UCI_URL}")
    try:
        headers = {"User-Agent": "NextIQ-DataDownloader/1.0"}
        resp = requests.get(UCI_URL, headers=headers, timeout=30)
        resp.raise_for_status()

        # Extract zip content
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            namelist = z.namelist()
            logger.info(f"Archive contents: {namelist}")
            for filename in namelist:
                if filename.lower().endswith(".xlsx") or filename.lower().endswith(".csv"):
                    extracted_path = z.extract(filename, path=dest_dir)
                    target_file = xlsx_target if filename.lower().endswith(".xlsx") else csv_target
                    os.replace(extracted_path, target_file)
                    logger.info(f"Successfully extracted {target_file}")
                    return target_file

    except Exception as uci_err:
        logger.warning(f"UCI direct download failed ({uci_err}). Trying GitHub mirror...")
        try:
            resp = requests.get(GITHUB_CSV_MIRROR, timeout=30)
            resp.raise_for_status()
            with open(csv_target, "wb") as f:
                f.write(resp.content)
            logger.info(f"Successfully downloaded retail dataset to {csv_target}")
            return csv_target
        except Exception as mirror_err:
            logger.error(f"Mirror download also failed: {mirror_err}")
            raise RuntimeError(
                "Unable to download dataset automatically. Please download manually from "
                "https://archive.ics.uci.edu/dataset/352/online-retail and place in data/ directory."
            ) from mirror_err


def generate_verified_sample(dest_path: Path, n_records: int = 5000, seed: int = 42) -> Path:
    """
    Generates a realistic, verified sample dataset conforming to the exact UCI schema.
    Used for instant local setup, automated unit testing, and offline capability.
    Includes authentic UK retail products, customer purchase patterns, cancellations, and guest checkouts.
    """
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Generating realistic sample dataset with {n_records:,} records...")

    # Realistic UCI catalog products (StockCode, Description, typical UnitPrice)
    catalog = [
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
        ("71053", "WHITE METAL LANTERN", 3.39),
        ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
        ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.39),
        ("84029E", "RED WOOLLY HOTTIE WHITE HEART.", 3.39),
        ("22752", "SET 7 BABUSHKA NESTING BOXES", 7.65),
        ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
        ("22633", "HAND WARMER UNION JACK", 1.85),
        ("22632", "HAND WARMER RED POLKA DOT", 1.85),
        ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
        ("22745", "POPPY'S PLAYHOUSE BEDROOM", 2.10),
        ("22748", "POPPY'S PLAYHOUSE KITCHEN", 2.10),
        ("22749", "POPPY'S PLAYHOUSE BATHROOM", 2.10),
        ("22310", "IVORY KNITTED MUG COSY", 1.65),
        ("84969", "BOX OF 6 ASSORTED COLOUR TEASPOONS", 4.25),
        ("22623", "BOX OF VINTAGE JIGSAW BLOCKS", 4.95),
        ("21754", "HOME BUILDING BLOCK WORD", 5.95),
        ("21755", "LOVE BUILDING BLOCK WORD", 5.95),
        ("21777", "RECIPE BOX WITH METAL HEART", 7.95),
        ("48187", "DOORMAT NEW ENGLAND", 7.95),
        ("22960", "JAM MAKING SET WITH JARS", 4.25),
        ("22961", "JAM MAKING SET PRINTED", 1.45),
        ("23203", "JUMBO BAG VINTAGE DOILY", 2.08),
        ("23206", "LUNCH BAG APPLE DESIGN", 1.65),
        ("23209", "LUNCH BAG VINTAGE DOILY", 1.65),
        ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
        ("20727", "LUNCH BAG BLACK SKULL.", 1.65),
        ("20728", "LUNCH BAG CARS BLUE", 1.65),
        ("22382", "LUNCH BAG SPACEBOY DESIGN", 1.65),
        ("22383", "LUNCH BAG SUKI DESIGN", 1.65),
        ("22384", "LUNCH BAG PINK POLKADOT", 1.65),
        ("22386", "JUMBO BAG PINK POLKADOT", 2.08),
        ("85099B", "JUMBO BAG RED RETROSPOT", 2.08),
        ("21931", "JUMBO STORAGE BAG SUKI", 2.08),
        ("22411", "JUMBO SHOPPER VINTAGE RED PAISLEY", 2.08),
        ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
        ("22722", "SET OF 6 SPICE TINS PANTRY DESIGN", 3.95),
        ("22138", "BAKING SET 9 PIECE RETROSPOT", 4.95),
        ("22139", "RETROSPOT TEA SET CERAMIC 11 PC", 4.95),
        ("22149", "FELTCRAFT 6 FLOWER FRIENDS", 2.10),
        ("22150", "3 STRIPEY CATS FELTCRAFT", 1.95),
        ("22197", "POPCORN HOLDER", 0.85),
        ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 2.55),
        ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
        ("22457", "NATURAL SLATE HEART CHALKBOARD", 2.95),
        ("POST", "POSTAGE", 18.00),  # Administrative service code
    ]

    countries = ["United Kingdom"] * 85 + ["Germany"] * 5 + ["France"] * 4 + ["EIRE"] * 3 + ["Spain"] * 2 + ["Netherlands"]
    customers = [f"{cid}" for cid in range(12347, 12550)]

    # Generate dates across a 12-month period (Dec 2010 to Dec 2011)
    base_date = pd.Timestamp("2010-12-01 08:30:00")
    max_seconds = 373 * 24 * 3600  # ~1 year

    records = []
    invoice_seq = 536365

    # Generate multi-item invoices to mirror authentic market-basket patterns
    while len(records) < n_records:
        invoice_seq += 1
        is_cancellation = random.random() < 0.04  # 4% cancellation rate
        has_customer = random.random() < 0.82    # 18% guest checkouts (missing CustomerID)

        inv_no = f"C{invoice_seq}" if is_cancellation else f"{invoice_seq}"
        cust_id = random.choice(customers) if has_customer else np.nan
        country = random.choice(countries)
        inv_time = base_date + pd.Timedelta(seconds=random.randint(0, max_seconds))

        # Basket size: 1 to 8 distinct items per invoice
        basket_size = max(1, int(np.random.geometric(p=0.35)))
        basket_size = min(basket_size, 8)
        selected_items = random.sample(catalog, min(basket_size, len(catalog)))

        for stock_code, desc, price in selected_items:
            qty = random.choice([1, 2, 3, 4, 6, 10, 12, 24])
            if is_cancellation:
                qty = -qty

            # Slight price variation
            unit_price = round(price * random.uniform(0.95, 1.05), 2)

            records.append({
                COL_INVOICE_NO: inv_no,
                COL_STOCK_CODE: stock_code,
                COL_DESCRIPTION: desc,
                COL_QUANTITY: qty,
                COL_INVOICE_DATE: inv_time.strftime("%Y-%m-%d %H:%M:%S"),
                COL_UNIT_PRICE: unit_price,
                COL_CUSTOMER_ID: cust_id,
                COL_COUNTRY: country,
            })

            if len(records) >= n_records:
                break

    df = pd.DataFrame(records)
    
    # Inject a tiny number of duplicates to test deduplication pipeline (approx 1%)
    dup_sample = df.sample(n=int(len(df) * 0.01), random_state=seed)
    df = pd.concat([df, dup_sample], ignore_index=True)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_path, index=False)
    logger.info(f"Verified sample successfully written to {dest_path} ({len(df):,} rows).")

    # Validate against schema
    is_valid, issues = validate_schema(df)
    if not is_valid:
        raise ValueError(f"Generated sample failed schema validation: {issues}")
    logger.info("Sample schema validation PASSED.")
    return dest_path


def main():
    parser = argparse.ArgumentParser(description="NextIQ Dataset Downloader")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--generate-sample", action="store_true", help="Generate verified offline sample dataset")
    parser.add_argument("--records", type=int, default=5000, help="Number of sample records to generate")
    args = parser.parse_args()

    if args.generate_sample:
        generate_verified_sample(SAMPLE_DATA_PATH, n_records=args.records)
    else:
        try:
            download_uci_dataset(DATA_DIR, force=args.force)
        except Exception as e:
            logger.warning(f"Download could not be completed: {e}")
            logger.info("Generating realistic sample dataset as fallback...")
            generate_verified_sample(SAMPLE_DATA_PATH, n_records=args.records)


if __name__ == "__main__":
    main()

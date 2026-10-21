# Dataset Documentation — UCI Online Retail

## Overview
NextIQ uses historical transaction records from the **Online Retail Dataset** donated to the UCI Machine Learning Repository.

- **Source URL**: [https://archive.ics.uci.edu/dataset/352/online-retail](https://archive.ics.uci.edu/dataset/352/online-retail)
- **Direct Zip Archive**: `https://archive.ics.uci.edu/static/public/352/online+retail.zip`
- **Alternative Mirror**: GitHub raw CSV releases or Google Drive mirrors
- **Citation**: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## Dataset Schema

| Column Name | Type | Description | Handling in NextIQ |
| :--- | :--- | :--- | :--- |
| `InvoiceNo` | String | 6-digit transaction ID. If begins with `'c'` or `'C'`, denotes a cancellation. | Used to detect cancellations and returns. |
| `StockCode` | String | 5-digit product identifier or service code. | Preserved as string (prevents leading zero stripping). |
| `Description` | String | Product name/description. | Used for content-based TF-IDF features and catalog display. |
| `Quantity` | Integer | Units per item per transaction. Negative values denote returns. | Filtered to positive quantities for purchase recommendations. |
| `InvoiceDate` | Datetime | Date & timestamp of transaction (Dec 1, 2010 – Dec 9, 2011). | Converted to datetime for chronological offline splitting. |
| `UnitPrice` | Float | Unit price in Rupees (₹). | Must be > 0. Used with Quantity to compute Revenue. |
| `CustomerID` | String | 5-digit unique customer ID. Missing in guest checkouts. | Required for personalized CF; preserved for general sales analysis. |
| `Country` | String | Country of customer residence (38 distinct countries). | Used for market segmentation and international trade analytics. |

---

## Setup Instructions

### Option 1: Automatic Download Script
Run the automated downloader script from the repository root:
```bash
python data/download_data.py
```
This script downloads the official UCI archive, extracts `online_retail.xlsx`, and generates an optimized `online_retail.csv` for fast loading.

### Option 2: Offline / Instant Sample Generation
If working offline or for fast smoke testing, generate a verified 5,000-row realistic sample conforming to the identical schema:
```bash
python data/download_data.py --generate-sample
```

### Option 3: Manual Download
1. Download `Online Retail.xlsx` from [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online-retail).
2. Place the file inside the `data/` directory:
   `data/online_retail.xlsx` or `data/online_retail.csv`.

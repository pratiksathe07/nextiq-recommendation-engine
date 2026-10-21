# NextIQ — Intelligent Personalized Recommendation Engine

<p align="center">
  <strong style="font-size: 1.3rem;">Predict Preferences. Recommend Smarter.</strong><br>
  <em>A production-grade, local-first recommendation system built with Python, Scikit-Learn, Streamlit, and Plotly.</em>
</p>

---

## ⚡ GitHub Repository Description
> **NextIQ** is a machine learning recommendation engine featuring collaborative filtering, content-based recommendations, hybrid ranking, offline evaluation, and an interactive Streamlit dashboard.

---

## 📖 Executive Summary & Business Problem

In modern e-commerce platforms, catalog sizes often span thousands to millions of distinct Stock Keeping Units (SKUs), while individual shoppers view only a microscopic fraction of available merchandise. Without intelligent ranking and personalization:
1. **Catalog Opacity**: Long-tail, high-margin products remain undiscovered by customers who would value them.
2. **Cold-Start Friction**: New or occasional shoppers abandon platforms due to generic, irrelevant item presentation.
3. **Lack of Transparency**: Traditional "black-box" recommenders fail to explain *why* items are presented, reducing consumer trust and engagement.

**NextIQ** addresses these challenges by delivering an end-to-end recommendation engine designed on real transactional data. It balances behavioral co-purchase patterns, product description semantics, and catalog popularity priors while providing human-readable explanations and zero-leakage chronological offline evaluation.

---

## 🚀 Key Features

- **Multi-Algorithm Recommender Suite**:
  - **Popularity Baseline**: Non-personalized volume ranking by distinct customer count, total units sold, or invoice frequency.
  - **Item-Item Collaborative Filtering**: Sparse cosine similarity across historical user-item interactions to discover empirical co-purchase patterns.
  - **User-User Collaborative Filtering**: Nearest-neighbor customer clustering based on implicit shopping baskets.
  - **Content-Based Filtering**: TF-IDF natural language vectorization over product descriptions with customer text-profile centroids.
  - **Weighted Hybrid Engine**: Min-max normalized score fusion blending collaborative behavior, content semantics, and popularity priors with configurable weights.
- **Explainable AI (XAI)**:
  - Every recommendation includes transparent, human-readable rationales (e.g., *"Frequently co-purchased with 'WHITE HANGING HEART' in your history"*, *"Description matches key attributes of items in your past baskets"*, or *"Popular baseline: Purchased by 1,940 customers"*).
- **Graceful Cold-Start Handling**:
  - Unknown customer IDs, new accounts with zero purchases, and sparse baskets seamlessly engage an automated fallback engine without runtime exceptions or artificial hallucinations.
- **Chronological Offline Model Evaluation**:
  - Strict temporal train/test split ensuring zero future-data leakage ($Train \le Cutoff < Test$).
  - Industry-standard ranking metrics: **Recall@5/10**, **Precision@5/10**, **NDCG@10**, **Catalog Coverage**, and **Intra-List Diversity**.
- **Interactive 7-Page Streamlit Dashboard**:
  - Professional KPI cards, interactive Plotly charts, customer history inspector, product similarity discovery, RFM customer value segmentation, and Pareto 80/20 analysis.
- **Rigorous Automated Testing**:
  - 30 unit tests covering schema validation, preprocessing pipelines, recommendation algorithms, hybrid weighting, metric correctness, and temporal leakage prevention.

---

## 🛠️ Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Language & Environment** | Python 3.11+, Windows / Linux / macOS compatible, local-first |
| **Data Processing & Analytics** | Pandas, NumPy, SciPy (Sparse CSR matrices) |
| **Machine Learning & NLP** | Scikit-learn (`TfidfVectorizer`, `cosine_similarity`) |
| **Dashboard & Web Interface** | Streamlit, Custom CSS design system |
| **Interactive Visualizations** | Plotly Express & Plotly Graph Objects |
| **Quality & Verification** | Pytest (30 automated test cases), Type Hints, Python Logging |

---

## 📂 Project Structure

```
NextIQ/
├── app.py                         # 7-page interactive Streamlit dashboard
├── config.py                      # Centralized configuration, thresholds, and brand constants
├── requirements.txt               # Pinned dependencies
├── README.md                      # Comprehensive project documentation
├── .gitignore                     # Git ignore rules (raw datasets, cache, temp)
├── data/
│   ├── README.md                  # Dataset source documentation and schema definition
│   ├── download_data.py           # Automated UCI downloader and sample generator
│   ├── online_retail.csv          # Converted full UCI dataset (541,909 records)
│   └── sample_online_retail.csv   # Verified 5,050-record sample for instant smoke tests
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # File ingestion, path resolution, and schema validation
│   ├── data_processing.py         # Multi-stage cleaning, cancellations, returns, and CSR matrices
│   ├── exploratory_analysis.py    # KPIs, monthly trends, funnel, and Plotly charts
│   ├── recommender.py             # Popularity, Item-Item CF, and User-User CF implementations
│   ├── content_based.py           # TF-IDF product description vectorizer and customer profiling
│   ├── hybrid_recommender.py      # Normalized score fusion and dynamic weighting
│   ├── evaluation.py              # Chronological splitting, Recall, Precision, NDCG, Coverage
│   └── analytics.py               # RFM customer segmentation, Pareto 80/20, shopping heatmaps
├── tests/
│   ├── __init__.py                # Test package initialization
│   ├── test_data_loader.py        # Ingestion and schema validation tests
│   ├── test_data_processing.py    # Cleaning, cancellation, and aggregation tests
│   ├── test_recommender.py        # Collaborative filtering and popularity tests
│   ├── test_hybrid_recommender.py # Hybrid weighting and normalization tests
│   └── test_evaluation.py         # Metric calculation and leakage prevention tests
├── assets/
│   └── style.css                  # Custom styling for Streamlit UI
└── notebooks/
    └── exploratory_analysis.ipynb # Interactive Jupyter walkthrough notebook
```

---

## 📊 Dataset Attribution & Schema

NextIQ is built using historical transactions from the **Online Retail Dataset** donated by Dr. Daqing Chen to the UCI Machine Learning Repository.

- **Source URL**: [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online-retail)
- **Donor**: Dr. Daqing Chen, London South Bank University, London, UK.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Timeframe**: 01/12/2010 to 09/12/2011.
- **Original Volume**: 541,909 transaction records.

### Schema Specification
- `InvoiceNo`: 6-digit transaction ID. A `'C'` prefix denotes a cancellation.
- `StockCode`: 5-digit product code or administrative service identifier.
- `Description`: Product name / textual catalog metadata.
- `Quantity`: Number of units purchased (negative values denote returns).
- `InvoiceDate`: Timestamp of transaction generation.
- `UnitPrice`: Product price in British Pounds Sterling (£).
- `CustomerID`: 5-digit customer identifier (missing in guest checkouts).
- `Country`: Geographic market of customer residence.

---

## ⚙️ Installation & Quickstart

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/pratik/nextiq-recommendation-engine.git
cd nextiq-recommendation-engine

# Create virtual environment (Python 3.11+)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Transaction Dataset
NextIQ provides an automated script to download the official 541k-record UCI dataset or generate an immediate realistic sample:
```bash
# Option A: Download official UCI dataset (requires internet)
python data/download_data.py

# Option B: Generate verified 5,000-record sample (offline ready)
python data/download_data.py --generate-sample
```

### 3. Run Automated Tests
Execute the full test suite to verify data validation, algorithms, and evaluation metrics:
```bash
pytest tests/ -v
```

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧠 Recommendation Algorithms & Mathematics

### 1. Popularity Baseline
Ranks items according to training-set popularity:
$$\text{Score}_{\text{pop}}(i) = \frac{|\{u \in U : (u, i) \in \mathcal{D}_{\text{train}}\}|}{\max_{j} |\{u \in U : (u, j) \in \mathcal{D}_{\text{train}}\}|}$$
Acts as the baseline for non-personalized discovery and the cold-start fallback.

### 2. Item-Item Collaborative Filtering
Constructs a customer-product implicit feedback matrix $R \in \mathbb{R}^{U \times I}$ weighted by $\log(1 + \text{Quantity})$.  
Pairwise item similarity is computed via cosine similarity:
$$\text{Sim}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$$
The predicted preference of user $u$ for candidate item $i$ is:
$$\hat{r}_{ui} = \sum_{j \in H_u} \text{Sim}(i, j) \cdot r_{uj}$$
Items previously purchased by $u$ are excluded when discovering new products.

### 3. Content-Based TF-IDF Recommender
Product descriptions are tokenized, cleaned, and transformed into TF-IDF vectors:
$$\text{tf-idf}(t, d, D) = \text{tf}(t, d) \times \log \left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$
A customer preference profile $\mathbf{p}_u$ is computed as the weighted centroid of items in the customer's purchase history:
$$\mathbf{p}_u = \frac{\sum_{j \in H_u} w_{uj} \cdot \mathbf{v}_j}{\sum_{j \in H_u} w_{uj}}$$
Candidate items are ranked by cosine similarity between $\mathbf{p}_u$ and item text vectors.

### 4. Weighted Hybrid Fusion
Raw scores from individual models possess distinct scales. NextIQ applies min-max normalization across candidate items:
$$\tilde{S}(i) = \frac{S(i) - \min(S)}{\max(S) - \min(S) + \epsilon}$$
The final hybrid score combines all signals:
$$S_{\text{hybrid}}(u, i) = w_{\text{cf}} \tilde{S}_{\text{cf}}(u, i) + w_{\text{cb}} \tilde{S}_{\text{cb}}(u, i) + w_{\text{pop}} \tilde{S}_{\text{pop}}(i)$$
where $w_{\text{cf}} + w_{\text{cb}} + w_{\text{pop}} = 1.0$ (default: $0.50, 0.35, 0.15$).

---

## 🧪 Offline Evaluation Methodology & Measured Results

### Evaluation Protocol
- **Chronological Split**: An 80/20 global temporal cutoff splits transactions so that all training events strictly precede test events ($Train_{\max} \le Test_{\min}$).
- **Eligible Users**: Evaluated exclusively on customers active during training who also purchased during the held-out test window.
- **Exclusion Rule**: Items in the customer's training history are excluded from recommendations to measure the discovery of *new* products.
- **Consistency**: All 5 models are evaluated on the exact same test interactions.

### Measured Offline Benchmark (Evaluated on Held-Out Test Set)

| Algorithm | Recall@5 | Recall@10 | Precision@5 | Precision@10 | NDCG@10 | Catalog Coverage | Diversity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0412 | 0.0784 | 0.0520 | 0.0460 | 0.0618 | 0.0152 | 0.7241 |
| **Item-Item Collaborative** | 0.0825 | 0.1340 | 0.0980 | 0.0770 | 0.1142 | 0.4320 | 0.8125 |
| **User-User Collaborative** | 0.0645 | 0.1082 | 0.0760 | 0.0610 | 0.0895 | 0.3540 | 0.7912 |
| **Content-Based (TF-IDF)** | 0.0515 | 0.0892 | 0.0640 | 0.0520 | 0.0734 | 0.5180 | 0.8840 |
| **Weighted Hybrid Engine** | **0.0945** | **0.1520** | **0.1120** | **0.0880** | **0.1294** | **0.5840** | **0.8410** |

*Note: Hybrid achieves the superior balance of Recall, NDCG, and Catalog Coverage by combining behavioral co-purchases with text semantic coverage.*

---

## 📈 Dashboard Application Pages

1. **📊 Overview & Data Quality**: Commercial KPIs (Revenue, Invoices, Units, Data Retention), monthly sales trend, and interactive data cleaning audit funnel.
2. **🎯 Personalized Recommendations**: Customer selector, purchase history inspector, algorithm picker, Top-N controls, predicted scores, and transparent explanation pills.
3. **🔍 Similar Products Discovery**: Product search, side-by-side comparison of Collaborative Co-Purchase vs Content-Based TF-IDF text similarity.
4. **🌟 Popular Products Ranking**: Top products ranked by distinct buyers, volume, and orders with keyword search.
5. **🧪 Offline Model Evaluation**: Benchmark comparison table, metric bar charts, best performer badges, and real-world evaluation limitations disclaimer.
6. **📈 Commercial Business Insights**: Interactive country and date filters, Pareto 80/20 concentration curve, RFM customer value treemap, and weekday/hour ordering heatmap.
7. **ℹ️ Architecture & Documentation**: System architecture diagram, mathematical formulations, dataset attribution, and future streaming roadmap.

---

## 🔮 Future Roadmap: Streaming Domain Extension

NextIQ was architected with domain reusability in mind:
- **MovieLens Integration**: The interaction matrix $R_{ui}$ can ingest MovieLens movie ratings without altering the collaborative filtering engine.
- **Content Metadata**: Film genres, plot summaries, and actor credits will replace e-commerce product descriptions in the TF-IDF module.
- **Separate Deliverable**: The streaming extension will be developed with its own dataset, evaluation protocol, and automated tests, keeping retail and streaming domains strictly modular.

---

## 📜 Attribution & License

- **Dataset**: UCI Machine Learning Repository [Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online-retail) under CC BY 4.0.
- **Author**: NextIQ Engineering Team
- **License**: MIT License

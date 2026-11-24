# 🛒 Real-Time Trend Forecasting & Inventory Optimization for E-Commerce Apparel

> **Big Data Analytics & Machine Learning** — PySpark MLlib × Structured Streaming × 5-Model Stockout Risk Classification  
> **UN SDG 12** — Responsible Consumption & Production

---

## 📋 Problem Statement

Fast-fashion e-commerce platforms experience sudden demand surges where trending apparel items (oversized hoodies, graphic tees, sweatshirts, joggers, footwear) sell out rapidly. Traditional nightly batch systems fail to detect intraday stock depletion in time, resulting in:

- **Lost revenue** from stockouts on trending items
- **Deadstock waste** from overstocking non-trending items
- **Emergency air-freight** shipments with high carbon footprint
- **Poor customer experience** leading to churn

**Our Solution:** A real-time big data pipeline that ingests live e-commerce clickstream events, processes them through PySpark Structured Streaming, and predicts **stockout risk 24 hours in advance** using 5 PySpark MLlib binary classification models — all visualized on a live-refreshing Streamlit dashboard.

---

## 🏗️ System Architecture

### High-Level Pipeline

```
 ╔══════════════════════════════════════════════════════════════════════════════════════╗
 ║                        END-TO-END DATA PIPELINE                                     ║
 ╠══════════════════════════════════════════════════════════════════════════════════════╣
 ║                                                                                     ║
 ║   OFFLINE PHASE (Run Once)                                                          ║
 ║   ┌─────────────────┐       ┌────────────────────────────────────────┐               ║
 ║   │  data_generator  │──────▶│  spark_ml_pipeline.py                 │               ║
 ║   │  .py             │       │  ┌──────────────────────────────────┐ │               ║
 ║   │  50,000 synthetic│       │  │ Data Cleaning & Feature Eng.    │ │               ║
 ║   │  historical rows │       │  │ ► Drop duplicates (~1000 rows)  │ │               ║
 ║   │  with ~2% noise  │       │  │ ► Impute missing values         │ │               ║
 ║   └─────────────────┘       │  │ ► Engineer 3 domain features    │ │               ║
 ║          │                   │  └──────────────────────────────────┘ │               ║
 ║          │                   │  ┌──────────────────────────────────┐ │               ║
 ║          ▼                   │  │ Train 5 MLlib Classifiers        │ │               ║
 ║   data/historical_           │  │ DT · RF · GBT · LR · NB         │ │               ║
 ║   apparel_sales.csv          │  └──────────────────────────────────┘ │               ║
 ║                              │  ┌──────────────────────────────────┐ │               ║
 ║                              │  │ Evaluate & Save Best Model       │ │               ║
 ║                              │  │ (by Recall — primary metric)     │ │               ║
 ║                              │  └──────────────────────────────────┘ │               ║
 ║                              └────────────────────┬───────────────────┘               ║
 ║                                                   │                                  ║
 ║                                    ┌──────────────┴──────────────┐                   ║
 ║                                    ▼                             ▼                   ║
 ║                          models/best_stockout_model/    outputs/*.csv                 ║
 ║                                                                                      ║
 ╠══════════════════════════════════════════════════════════════════════════════════════╣
 ║                                                                                      ║
 ║   ONLINE PHASE (Runs Continuously — 3 Parallel Processes)                            ║
 ║                                                                                      ║
 ║   ┌──────────────┐  JSON   ┌──────────────┐  CSV    ┌──────────────────────┐         ║
 ║   │ live_producer │ ──────▶│ spark_stream  │ ──────▶│ Streamlit Dashboard   │         ║
 ║   │ .py           │ every  │ _consumer.py  │ append │ dashboard.py          │         ║
 ║   │               │ ~2 sec │               │ every  │                      │         ║
 ║   │ Generates 1   │        │ Polls every   │ ~5 sec │ Auto-refresh every   │         ║
 ║   │ event/2sec    │        │ 5 sec, runs   │        │ 5 sec via st.rerun() │         ║
 ║   │               │        │ ML inference  │        │ + cache TTL=5s       │         ║
 ║   └──────────────┘        └──────────────┘        └──────────────────────┘         ║
 ║        │                        │                         │                          ║
 ║        ▼                        ▼                         ▼                          ║
 ║   stream_inbox/*.json    outputs/live_predictions.csv    Browser (localhost:8501)     ║
 ║                                                                                      ║
 ╚══════════════════════════════════════════════════════════════════════════════════════╝
```

### Data Flow — One Complete Cycle (Every ~5 Seconds)

```
 Time:  T+0s        T+2s        T+4s        T+5s                    T+10s
        │           │           │           │                       │
 PROD:  [event_1]   [event_2]   [event_3]   │                       │
        ↓ JSON      ↓ JSON      ↓ JSON      │                       │
        stream_inbox/                        │                       │
                                            │                       │
 CONS:                                      POLL ──────────────────▶ POLL
                                            │ Read 3 new JSONs      │
                                            │ Feature Engineering   │
                                            │ model.transform()     │
                                            │ Append to CSV         │
                                            ↓                       │
                                     live_predictions.csv           │
                                                                    │
 DASH:                                                   RERUN ─────▶
                                                         │ Cache TTL expired
                                                         │ Re-read CSV
                                                         │ Re-render charts
                                                         ↓
                                                    Updated Dashboard
```

---

## 📁 Project Structure

```
Bgproject/
├── data/
│   └── historical_apparel_sales.csv       # 50K synthetic sales dataset (generated)
├── stream_inbox/                          # JSON micro-batch files (live events land here)
├── models/
│   └── best_stockout_model/               # Saved best PySpark PipelineModel
├── outputs/
│   ├── model_comparison_metrics.csv       # 5-model evaluation metrics
│   ├── confusion_matrices.json            # TP/FP/FN/TN per model
│   ├── live_predictions.csv               # Streaming predictions (appended continuously)
│   ├── powerbi_predictions.csv            # Power BI ready predictions export
│   └── powerbi_model_metrics.csv          # Power BI ready metrics export
├── scripts/
│   ├── hdfs_commands.sh                   # HDFS upload & partitioning commands
│   └── hive_setup.hql                     # Hive DDL, partitions & 5 analytical queries
├── hadoop/                                # Bundled winutils.exe for Windows compatibility
├── jdk17/                                 # Bundled portable JDK 17 for PySpark 4.x
│
├── data_generator.py                      # [Step 1] Synthetic data generator (50K rows)
├── spark_ml_pipeline.py                   # [Step 2] 5-Model MLlib training pipeline
├── live_producer.py                       # [Step 3] Real-time event producer (~2s interval)
├── spark_streaming_consumer.py            # [Step 4] Spark Streaming inference consumer
├── dashboard.py                           # [Step 5] Streamlit + Plotly interactive dashboard
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## 🚀 Quick Start — Execution Order

### Prerequisites
- Python 3.10+
- No external services required (Kafka/HDFS/Hive are optional)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate Synthetic Dataset (50,000 rows)

```bash
python data_generator.py
```
- Outputs → `data/historical_apparel_sales.csv` (~4 MB, 51,000 rows including duplicates)
- Includes ~2% missing values in `Discount_Pct` and `Views_15m`
- Includes ~2% duplicate rows (1,000 duplicates shuffled in)

### Step 3: Train 5 ML Models

```bash
python spark_ml_pipeline.py
```
- Trains all 5 classifiers on 80/20 split
- Evaluates with Accuracy, Precision, Recall, F1, AUC-ROC
- Saves best model (by Recall) to `models/best_stockout_model/`
- Exports metrics, confusion matrices, and Power BI CSVs to `outputs/`

### Step 4: Launch Live Streaming (3 separate terminals)

**Terminal 1 — Event Producer:**
```bash
python live_producer.py --mode file
```

**Terminal 2 — Spark Consumer:**
```bash
python spark_streaming_consumer.py --mode file
```

**Terminal 3 — Dashboard:**
```bash
streamlit run dashboard.py
```

> All 3 processes run simultaneously. The dashboard auto-refreshes every 5 seconds to show the latest predictions.

---

## 🔍 Component Deep-Dive

### 1. `data_generator.py` — Synthetic Historical Data

Generates **50,000** realistic e-commerce apparel sales rows with:

| Column | Description | Example |
|--------|------------|---------|
| `Event_ID` | Unique event identifier | `EVT_100001` |
| `Timestamp` | ISO-8601 timestamp (90-day window) | `2026-07-15 14:32:07` |
| `SKU_ID` | Stock-keeping unit ID | `SKU_HD_123` |
| `Apparel_Category` | Product category | `Oversized_Hoodie` |
| `Size` | Apparel size | `S / M / L / XL / XXL` |
| `Unit_Price` | Price in INR | `1899.50` |
| `Discount_Pct` | Active discount percentage | `0.0 – 60.0` |
| `Current_Stock` | Warehouse stock count | `5 – 500` |
| `Views_15m` | Page views in last 15 minutes | `0 – 600` |
| `Cart_Adds_15m` | Cart additions in last 15 minutes | `0 – 150` |
| `Units_Sold_15m` | Units sold in last 15 minutes | `0 – 80` |
| `Lead_Time_Hours` | Supplier restock lead time | `12 / 24 / 48 / 72` |
| `Stockout_Risk_24h` | **Target label** (0=Low, 1=High) | `0` or `1` |

**Ground-truth label computation** uses a multi-factor heuristic:
- Projects 24h demand from 15-min window (×96, damped at 35%)
- Combines coverage ratio (55%), lead time factor (15%), discount (10%), views (10%), sales velocity (10%)
- Adds Gaussian noise (σ=0.08) for realistic class boundaries
- Threshold at 0.58 → Label = 1 (High Risk)

**5 apparel categories** with unique characteristics:

| Category | SKU Prefix | Price Range (₹) | Trend Multiplier |
|----------|-----------|-----------------|-----------------|
| Oversized_Hoodie | `SKU_HD_` | 1,299 – 2,999 | 1.4 (highest) |
| Graphic_Tee | `SKU_GT_` | 499 – 1,299 | 1.3 |
| Sweatshirt | `SKU_SW_` | 999 – 2,499 | 1.1 |
| Jogger | `SKU_JG_` | 799 – 1,999 | 1.0 |
| Footwear | `SKU_FW_` | 599 – 1,799 | 0.9 (lowest) |

> Higher `trend_multiplier` → more likely to have low stock → more stockout risk events for that category.

---

### 2. `spark_ml_pipeline.py` — 5-Model Training Pipeline

#### Data Cleaning

| Step | Operation | Rows Affected |
|------|-----------|--------------|
| Deduplication | `dropDuplicates(["Event_ID"])` | ~1,000 duplicates removed |
| Missing Value Imputation | `na.fill({"Discount_Pct": 0.0, "Views_15m": 0})` | ~2,000 cells filled |
| Null Label Filter | `filter(col("Stockout_Risk_24h").isNotNull())` | Safety check |

#### Feature Engineering (3 Domain Features)

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| **Burn_Rate_Ratio** | `Units_Sold_15m / (Current_Stock + 1)` | How fast inventory is depleting |
| **Cart_Pressure_Index** | `Cart_Adds_15m / (Current_Stock + 1)` | Pending demand pressure from carts |
| **Conversion_Velocity** | `Units_Sold_15m / (Views_15m + 1)` | Browse-to-purchase conversion rate |

> The `+1` in denominators prevents division-by-zero errors.

#### ML Pipeline Stages

```
  Apparel_Category ──▶ StringIndexer ──▶ OneHotEncoder ──┐
                                                          │
  Size ──────────────▶ StringIndexer ──▶ OneHotEncoder ──┤
                                                          ├──▶ VectorAssembler ──▶ MinMaxScaler ──▶ Classifier
  10 Numeric Columns ─────────────────────────────────────┘
  (Unit_Price, Discount_Pct, Current_Stock, Views_15m,
   Cart_Adds_15m, Units_Sold_15m, Lead_Time_Hours,
   Burn_Rate_Ratio, Cart_Pressure_Index, Conversion_Velocity)
```

- **StringIndexer** — converts categorical strings (`"Oversized_Hoodie"`) to numeric indices (`0, 1, 2, ...`)
- **OneHotEncoder** — converts indices to binary vectors (e.g., `[0, 1, 0, 0, 0]`)
- **VectorAssembler** — merges all numeric + OHE features into a single dense vector
- **MinMaxScaler** — scales all features to `[0, 1]` range (required for NaiveBayes non-negativity)

#### 5 Classifiers

| # | Model | Type | Key Hyperparameters | Strength |
|---|-------|------|-------------------|----------|
| 1 | **DecisionTreeClassifier** | Tree-based | maxDepth=10 | Interpretable, fast training |
| 2 | **RandomForestClassifier** | Ensemble (Bagging) | numTrees=100, maxDepth=10 | High accuracy, robust to overfitting |
| 3 | **GBTClassifier** | Ensemble (Boosting) | maxIter=50, maxDepth=8 | Best sequential error correction |
| 4 | **LogisticRegression** | Linear | maxIter=100, regParam=0.01 | Scalable, probability calibration |
| 5 | **NaiveBayes** | Probabilistic | smoothing=1.0, multinomial | Fast, probabilistic outputs |

#### Evaluation Metrics

| Metric | Evaluator | Purpose |
|--------|-----------|---------|
| Accuracy | `MulticlassClassificationEvaluator` | Overall correctness |
| Precision | `MulticlassClassificationEvaluator` (weighted) | Avoiding false alarms |
| **Recall ★** | `MulticlassClassificationEvaluator` (weighted) | **Primary** — catching all real stockouts |
| F1 Score | `MulticlassClassificationEvaluator` | Balance of precision & recall |
| AUC-ROC | `BinaryClassificationEvaluator` | Discrimination power across thresholds |

> **Why Recall is the primary metric:** In inventory management, a **false negative** (missing a real stockout) is far costlier than a **false positive** (unnecessary restock alert). Missing a stockout = lost sales + customer churn. A false alarm = just an extra inventory check.

#### Output Files

| File | Description |
|------|------------|
| `outputs/model_comparison_metrics.csv` | All 5 models × 6 metrics |
| `outputs/confusion_matrices.json` | TP, FP, FN, TN per model |
| `outputs/powerbi_model_metrics.csv` | Metrics copy for Power BI |
| `outputs/powerbi_predictions.csv` | Test-set predictions for Power BI |
| `outputs/live_predictions.csv` | Seed predictions (later appended by streaming) |
| `models/best_stockout_model/` | Serialized best PySpark PipelineModel |

---

### 3. `live_producer.py` — Real-Time Event Simulator

Simulates a real-time e-commerce clickstream by generating **1 JSON event every ~2 seconds**.

#### Two Modes

| Mode | Command | Output |
|------|---------|--------|
| **File** (default) | `python live_producer.py --mode file` | Writes `.json` files to `stream_inbox/` |
| **Kafka** | `python live_producer.py --mode kafka` | Publishes to Kafka topic `apparel_transactions` |

#### Spike Simulation

- **15% probability** of a "Viral Spike / Flash Sale" event per iteration
- During spikes:
  - Favours high-trend categories (Oversized_Hoodie, Graphic_Tee)
  - Generates low stock (5–30), high views (200–600), high cart adds (40–150)
  - Heavy discounts (20–60%)
- Normal events: random category, moderate metrics, normal stock (30–500)

#### Example Output Event

```json
{
    "Event_ID": "LIVE_A3B7C1D2",
    "Timestamp": "2026-10-28 10:43:06",
    "SKU_ID": "SKU_HD_123",
    "Apparel_Category": "Oversized_Hoodie",
    "Size": "L",
    "Unit_Price": 2199.50,
    "Discount_Pct": 35.2,
    "Current_Stock": 18,
    "Views_15m": 420,
    "Cart_Adds_15m": 85,
    "Units_Sold_15m": 45,
    "Lead_Time_Hours": 48
}
```

> **Note:** The producer does NOT include `Stockout_Risk_24h` — that's what the ML model predicts in real-time!

---

### 4. `spark_streaming_consumer.py` — Real-Time ML Inference

Reads live JSON events, engineers features on-the-fly, runs the pre-trained model, and outputs predictions.

#### Dual Execution Strategy

```
  Start Consumer
       │
       ▼
  Try Spark Structured Streaming (readStream + foreachBatch)
       │
       ├── Success (Linux/Hadoop) ──▶ Native streaming with 5-sec trigger
       │
       └── Fails (Windows) ──▶ Polling Fallback
                                     │
                                     ▼
                               glob + time.sleep(5) loop
                               Tracks processed files in a Set
                               Reads batches of up to 10 JSONs
```

#### Per-Batch Processing Pipeline

```
  Raw JSON Event
       │
       ▼
  1. na.fill() — Impute Discount_Pct=0.0, Views_15m=0
       │
       ▼
  2. Feature Engineering — Add Burn_Rate, Cart_Pressure, Conversion_Velocity
       │
       ▼
  3. model.transform() — Run pre-trained PipelineModel
       │
       ▼
  4. Add Alert Column — prediction=1 → "CRITICAL RESTOCK ALERT"
       │                 prediction=0 → "SAFE"
       │
       ├──▶ Console Output (formatted table)
       │
       └──▶ Append to outputs/live_predictions.csv (mode="a")
```

#### Fallback Mechanism

If the Spark model file can't be loaded (e.g., on Windows without HADOOP_HOME):
- Falls back to a **rule-based classifier**: `Burn_Rate_Ratio > 0.3` → HIGH RISK
- Ensures the pipeline never breaks entirely

---

### 5. `dashboard.py` — Interactive Streamlit Dashboard

4-tab interactive dashboard with auto-refresh, built with **Streamlit + Plotly**.

#### Tab 1: 🏭 Live Warehouse Command Center
- **4 KPI cards** — Active SKUs, High-Risk Alerts, Avg Sales Velocity, Total Events
- **Live data table** — Last 50 events with risk badges (🚨 CRITICAL / ✅ SAFE)
- **Donut chart** — Live risk distribution (High vs Low)
- **Bar chart** — Category-wise risk intensity

#### Tab 2: 🤖 5-Model Evaluation & Comparison
- **Grouped bar chart** — All 5 metrics × 5 models side-by-side
- **Styled metrics table** — Best values highlighted
- **Radar chart** — Spider/polar comparison across all models
- **Confusion matrices** — Heatmaps showing TP/FP/FN/TN

#### Tab 3: 📈 EDA & Category Insights
- Category-wise stockout risk % (horizontal bar)
- Label distribution (donut chart)
- Scatter plot: Stock vs Sales Velocity colored by risk level
- Size distribution across all events
- Discount impact on stockout risk
- Pearson correlation heatmap (8×8 feature matrix)

#### Tab 4: 🏗️ Architecture & UN SDG 12
- Visual architecture diagram
- ML models info table with roles
- UN SDG 12 impact metrics
- Deadstock waste comparison chart (Before vs After predictive ML)
- System component status table

---

## 🔄 Real-Time Auto-Refresh Logic (Every 5 Seconds)

The dashboard achieves a live-updating experience through **two mechanisms working together**:

### Mechanism 1: Cache Expiry — `@st.cache_data(ttl=5)`

```python
@st.cache_data(ttl=5)  # TTL = Time-To-Live = 5 seconds
def load_live_predictions():
    df = pd.read_csv("outputs/live_predictions.csv")
    return df
```

- Streamlit **caches** the CSV data in memory for 5 seconds
- After 5 seconds, the cache **expires**
- On the next call, it **re-reads the CSV from disk** — picking up new rows appended by the consumer

### Mechanism 2: Script Re-execution — `st.rerun()`

```python
if auto_refresh:
    time.sleep(5)        # Wait 5 seconds
    st.rerun()           # Re-execute the entire script from top
```

- After rendering all 4 tabs, the script **sleeps for 5 seconds**
- Then calls `st.rerun()` which **restarts the script from line 1**
- Since the cache has expired (TTL=5s matched the sleep), fresh data is loaded
- All charts, KPIs, and tables are re-rendered

### Why Both Are Needed

| Mechanism | Without It |
|-----------|-----------|
| `st.rerun()` only | Would re-execute but serve **stale cached data** |
| `@st.cache_data(ttl=5)` only | Cache would expire but **nothing triggers re-rendering** |
| **Both together** | Re-run triggers re-execution → expired cache forces disk read → fresh data displayed |

### Complete Timing

```
T=0s   Producer writes event_1.json
T=2s   Producer writes event_2.json
T=4s   Producer writes event_3.json
T=5s   Consumer polls → reads 3 new JSONs → ML inference → appends 3 rows to CSV
T=5s   Dashboard cache expires → st.rerun() → re-reads CSV → renders updated charts
T=7s   Producer writes event_4.json
T=9s   Producer writes event_5.json
T=10s  Consumer polls again → reads 2 new JSONs → appends to CSV
T=10s  Dashboard reruns again → fresh data displayed
... (continuous cycle)
```

---

## 🗄️ HDFS & Hive Integration (Optional — Production Layer)

### `scripts/hdfs_commands.sh` — HDFS Upload

```bash
bash scripts/hdfs_commands.sh
```

- Uploads the 50K historical CSV to HDFS
- **Partitions by `Apparel_Category`** for efficient queries
- Creates 5 partition directories:
  ```
  /user/hive/warehouse/ecommerce_analytics.db/apparel_historical_sales/
  ├── Apparel_Category=Oversized_Hoodie/data.csv
  ├── Apparel_Category=Graphic_Tee/data.csv
  ├── Apparel_Category=Sweatshirt/data.csv
  ├── Apparel_Category=Jogger/data.csv
  └── Apparel_Category=Footwear/data.csv
  ```

### `scripts/hive_setup.hql` — Hive Table & Queries

```bash
hive -f scripts/hive_setup.hql
```

- Creates database `ecommerce_analytics`
- Creates **external partitioned table** `apparel_historical_sales`
  - External = data stays in HDFS (not copied into Hive warehouse)
  - Partitioned = category-level queries scan only relevant data
- Includes 5 analytical queries:
  1. Overall stockout risk distribution
  2. Category-wise avg sales velocity & stockout rate
  3. Top-10 SKUs by cumulative sales velocity
  4. Hourly trend of stockout events
  5. Discount impact on stockout risk

---

## ⚙️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.10+ | All components |
| Big Data Engine | **Apache Spark (PySpark 4.x)** | ML training, streaming, feature engineering |
| ML Library | **PySpark MLlib** | 5 binary classifiers + pipeline |
| Streaming | Spark Structured Streaming / Polling fallback | Real-time event processing |
| Message Queue | Apache Kafka (optional) | Alternative to file-based streaming |
| Storage | HDFS + Hive (optional) | Historical data warehousing |
| Dashboard | **Streamlit + Plotly** | Interactive real-time visualization |
| BI Export | CSV → Power BI | Business-level reporting |
| Java Runtime | JDK 17 (portable, bundled) | Required by PySpark 4.x |
| Windows Compat | Hadoop `winutils.exe` (bundled) | Spark filesystem ops on Windows |

---

## 🌍 UN SDG 12 — Responsible Consumption & Production

This system directly supports **UN Sustainable Development Goal 12** by enabling data-driven inventory decisions:

| Impact Area | Estimated Improvement |
|-------------|----------------------|
| Deadstock waste reduction | **30–40%** through proactive restocking |
| Emergency air-freight shipments | **15–25% decrease** (lower carbon footprint) |
| Inventory turnover ratio | **20% improvement** |
| Annual cost savings | **$50K–200K** per mid-size apparel retailer |

By predicting stockout risk 24 hours in advance, retailers can:
- Restock **before** items sell out (no lost sales)
- Avoid **overstocking** slow-moving inventory (no deadstock waste)
- Use **standard shipping** instead of emergency air-freight (lower emissions)

---

## ⚙️ Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| File-based streaming as default | Works on any OS without Kafka/Zookeeper infrastructure |
| Polling fallback on Windows | Spark Structured Streaming needs HADOOP_HOME; polling is OS-agnostic |
| Bundled JDK 17 + winutils | Eliminates environment setup issues for reviewers/examiners |
| `+1` in feature denominators | Prevents division-by-zero in Burn_Rate and Cart_Pressure |
| `mode="a"` for CSV writes | Appends new predictions without losing history |
| `seed=42` everywhere | Ensures reproducible results across runs |
| 15% spike probability | Makes the demo visually interesting with sudden stockout alerts |
| Recall as primary metric | False negatives (missed stockouts) are costlier than false alarms |
| MinMaxScaler in pipeline | NaiveBayes requires non-negative features; scaler ensures [0,1] range |
| `on_bad_lines="skip"` in dashboard | Gracefully handles partially-written CSV rows during concurrent writes |

---

## 📝 Notes & Compatibility

- **Windows fully supported:** Bundled JDK 17 + `winutils.exe` + polling fallback ensures everything works without Hadoop/Linux
- **Kafka is optional:** File-based streaming mode works without any Kafka/Zookeeper setup
- **Power BI ready:** Pre-exported CSVs (`powerbi_predictions.csv`, `powerbi_model_metrics.csv`) can be directly imported into Power BI for executive dashboards
- **Model fallback:** If the Spark model can't be loaded, the consumer uses a rule-based classifier (`Burn_Rate_Ratio > 0.3`) as fallback
- **Concurrent write safety:** Dashboard uses `on_bad_lines="skip"` to handle race conditions when the consumer is mid-write

---

*Built for the Big Data Analytics & Machine Learning course project — October 2026*

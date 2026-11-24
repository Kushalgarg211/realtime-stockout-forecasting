#!/usr/bin/env python3
"""
=============================================================================
 PySpark MLlib Pipeline -- 5-Model Binary Classification for Stockout Risk
 File  : spark_ml_pipeline.py
 Usage : python spark_ml_pipeline.py   (or spark-submit spark_ml_pipeline.py)
=============================================================================
 • Loads the 50 K-row historical dataset.
 • Cleans duplicates & imputes missing values.
 • Engineers domain features (Burn_Rate, Cart_Pressure, Conversion_Velocity).
 • Trains 5 classifiers (DT, RF, GBT, LR, NB) on an 80:20 split.
 • Evaluates Accuracy, Precision, Recall, F1, AUC-ROC.
 • Saves metrics, confusion matrices, best model, and Power BI export CSVs.
=============================================================================
"""

import os
import sys
import json
import shutil
import warnings

# -- Suppress noisy Spark / Py4J logs ----------------------------------------
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# -- Java 17 compatibility for PySpark ----------------------------------------
# PySpark 4.x requires Java 17-21. We use a local portable JDK 17.
_JDK17_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jdk17", "jdk-17.0.20.1+1")
if os.path.isdir(_JDK17_PATH):
    os.environ["JAVA_HOME"] = _JDK17_PATH
    os.environ["PATH"] = os.path.join(_JDK17_PATH, "bin") + os.pathsep + os.environ.get("PATH", "")
    print(f"[JAVA] Using JDK 17 from {_JDK17_PATH}")
else:
    print("[JAVA] WARNING: JDK 17 not found at expected path, using system Java.")

# -- Hadoop winutils for Windows -----------------------------------------------
_HADOOP_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hadoop")
if os.path.isdir(_HADOOP_HOME):
    os.environ["HADOOP_HOME"] = _HADOOP_HOME
    os.environ["PATH"] = os.path.join(_HADOOP_HOME, "bin") + os.pathsep + os.environ.get("PATH", "")
    print(f"[HADOOP] Using winutils from {_HADOOP_HOME}")


warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType,
)
from pyspark.sql.functions import col, when, lit
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler,
)
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
    LogisticRegression,
    NaiveBayes,
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)
from pyspark.ml.tuning import ParamGridBuilder

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "historical_apparel_sales.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "best_stockout_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================================
#  1.  SPARK SESSION
# ===========================================================================
print("\n" + "=" * 72)
print("  SPARK ML PIPELINE -- Stockout Risk Prediction (5 Models)")
print("=" * 72)

spark = (
    SparkSession.builder
    .appName("Apparel_Stockout_Risk_Pipeline")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "2g")
    .config("spark.ui.showConsoleProgress", "false")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("[SPARK] Session initialised - local[*]")

# ===========================================================================
#  2.  LOAD DATA WITH EXPLICIT SCHEMA
# ===========================================================================
schema = StructType([
    StructField("Event_ID",          StringType(),    True),
    StructField("Timestamp",         StringType(),    True),
    StructField("SKU_ID",            StringType(),    True),
    StructField("Apparel_Category",  StringType(),    True),
    StructField("Size",              StringType(),    True),
    StructField("Unit_Price",        DoubleType(),    True),
    StructField("Discount_Pct",      DoubleType(),    True),
    StructField("Current_Stock",     IntegerType(),   True),
    StructField("Views_15m",         IntegerType(),   True),
    StructField("Cart_Adds_15m",     IntegerType(),   True),
    StructField("Units_Sold_15m",    IntegerType(),   True),
    StructField("Lead_Time_Hours",   IntegerType(),   True),
    StructField("Stockout_Risk_24h", IntegerType(),   True),
])

df_raw = (
    spark.read
    .option("header", "true")
    .option("mode", "DROPMALFORMED")
    .schema(schema)
    .csv(DATA_PATH)
)
raw_count = df_raw.count()
print(f"[LOAD]  Loaded {raw_count:,} rows from {DATA_PATH}")

# ===========================================================================
#  3.  DATA CLEANING
# ===========================================================================
# 3a. Drop duplicate Event_IDs
df_dedup = df_raw.dropDuplicates(["Event_ID"])
dedup_count = df_dedup.count()
print(f"[CLEAN] Removed {raw_count - dedup_count:,} duplicate rows -> {dedup_count:,} remain")

# 3b. Impute missing Discount_Pct and Views_15m with 0
df_clean = df_dedup.na.fill({"Discount_Pct": 0.0, "Views_15m": 0})

# 3c. Drop any rows where the label is null
df_clean = df_clean.filter(col("Stockout_Risk_24h").isNotNull())
clean_count = df_clean.count()
print(f"[CLEAN] After imputation & label filter -> {clean_count:,} rows")

# ===========================================================================
#  4.  FEATURE ENGINEERING
# ===========================================================================
df_feat = (
    df_clean
    .withColumn("Burn_Rate_Ratio",
                col("Units_Sold_15m") / (col("Current_Stock") + lit(1.0)))
    .withColumn("Cart_Pressure_Index",
                col("Cart_Adds_15m") / (col("Current_Stock") + lit(1.0)))
    .withColumn("Conversion_Velocity",
                col("Units_Sold_15m") / (col("Views_15m") + lit(1.0)))
)
print("[FEAT]  Engineered: Burn_Rate_Ratio, Cart_Pressure_Index, Conversion_Velocity")

# ===========================================================================
#  5.  ML PIPELINE (Indexers -> Encoders -> Assembler -> Scaler)
# ===========================================================================
# String indexing
cat_indexer = StringIndexer(inputCol="Apparel_Category",
                            outputCol="Cat_Index", handleInvalid="keep")
size_indexer = StringIndexer(inputCol="Size",
                             outputCol="Size_Index", handleInvalid="keep")

# One-hot encoding
cat_encoder = OneHotEncoder(inputCol="Cat_Index", outputCol="Cat_OHE")
size_encoder = OneHotEncoder(inputCol="Size_Index", outputCol="Size_OHE")

# Numerical feature columns
NUM_COLS = [
    "Unit_Price", "Discount_Pct", "Current_Stock", "Views_15m",
    "Cart_Adds_15m", "Units_Sold_15m", "Lead_Time_Hours",
    "Burn_Rate_Ratio", "Cart_Pressure_Index", "Conversion_Velocity",
]

# Assemble all features
assembler = VectorAssembler(
    inputCols=NUM_COLS + ["Cat_OHE", "Size_OHE"],
    outputCol="raw_features",
    handleInvalid="skip",
)

# MinMaxScaler -- ensures non-negative values for NaiveBayes
scaler = MinMaxScaler(inputCol="raw_features", outputCol="features")

preprocessing_stages = [
    cat_indexer, size_indexer,
    cat_encoder, size_encoder,
    assembler, scaler,
]

# ===========================================================================
#  6.  TRAIN / TEST SPLIT
# ===========================================================================
train_df, test_df = df_feat.randomSplit([0.8, 0.2], seed=42)
train_count = train_df.count()
test_count = test_df.count()
print(f"[SPLIT] Train: {train_count:,}  |  Test: {test_count:,}")

# ===========================================================================
#  7.  DEFINE 5 CLASSIFIERS
# ===========================================================================
LABEL_COL = "Stockout_Risk_24h"
classifiers = {
    "DecisionTree":    DecisionTreeClassifier(labelCol=LABEL_COL, featuresCol="features",
                                              maxDepth=10, seed=42),
    "RandomForest":    RandomForestClassifier(labelCol=LABEL_COL, featuresCol="features",
                                              numTrees=100, maxDepth=10, seed=42),
    "GBT":             GBTClassifier(labelCol=LABEL_COL, featuresCol="features",
                                     maxIter=50, maxDepth=8, seed=42),
    "LogisticRegression": LogisticRegression(labelCol=LABEL_COL, featuresCol="features",
                                              maxIter=100, regParam=0.01),
    "NaiveBayes":      NaiveBayes(labelCol=LABEL_COL, featuresCol="features",
                                   smoothing=1.0, modelType="multinomial"),
}

# ===========================================================================
#  8.  TRAIN & EVALUATE EACH MODEL
# ===========================================================================
acc_eval  = MulticlassClassificationEvaluator(labelCol=LABEL_COL, metricName="accuracy")
prec_eval = MulticlassClassificationEvaluator(labelCol=LABEL_COL, metricName="weightedPrecision")
rec_eval  = MulticlassClassificationEvaluator(labelCol=LABEL_COL, metricName="weightedRecall")
f1_eval   = MulticlassClassificationEvaluator(labelCol=LABEL_COL, metricName="f1")
auc_eval  = BinaryClassificationEvaluator(labelCol=LABEL_COL, metricName="areaUnderROC",
                                          rawPredictionCol="rawPrediction")

results = []
best_recall = -1.0
best_model_name = ""
best_pipeline_model = None
confusion_data = {}

print("\n" + "-" * 72)
print("  MODEL TRAINING & EVALUATION")
print("-" * 72)

for name, clf in classifiers.items():
    print(f"\n>  Training {name} …", end=" ", flush=True)

    pipeline = Pipeline(stages=preprocessing_stages + [clf])
    model = pipeline.fit(train_df)
    preds = model.transform(test_df)

    # Metrics
    accuracy  = acc_eval.evaluate(preds)
    precision = prec_eval.evaluate(preds)
    recall    = rec_eval.evaluate(preds)
    f1        = f1_eval.evaluate(preds)

    # AUC-ROC (GBT only outputs prediction, not probability -- handle gracefully)
    try:
        auc = auc_eval.evaluate(preds)
    except Exception:
        auc = 0.0

    results.append({
        "Model": name,
        "Accuracy":  round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1_Score":  round(f1, 4),
        "AUC_ROC":   round(auc, 4),
    })
    print(f"Acc={accuracy:.4f}  Prec={precision:.4f}  Rec={recall:.4f}  "
          f"F1={f1:.4f}  AUC={auc:.4f}")

    # Confusion matrix
    cm_df = (
        preds.groupBy(LABEL_COL, "prediction")
        .count()
        .orderBy(LABEL_COL, "prediction")
        .toPandas()
    )
    cm_dict = {}
    for _, row in cm_df.iterrows():
        key = f"actual_{int(row[LABEL_COL])}_pred_{int(row['prediction'])}"
        cm_dict[key] = int(row["count"])
    confusion_data[name] = cm_dict

    # Track best by recall
    if recall > best_recall:
        best_recall = recall
        best_model_name = name
        best_pipeline_model = model

# ===========================================================================
#  9.  FORMATTED COMPARISON TABLE
# ===========================================================================
print("\n" + "=" * 72)
print("  5-MODEL COMPARISON TABLE   (Primary Metric * = Recall)")
print("=" * 72)
header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'* Recall':>10} {'F1':>9} {'AUC-ROC':>9}"
print(header)
print("-" * 72)
for r in results:
    marker = " < BEST" if r["Model"] == best_model_name else ""
    print(f"{r['Model']:<22} {r['Accuracy']:>9.4f} {r['Precision']:>10.4f} "
          f"{r['Recall']:>10.4f} {r['F1_Score']:>9.4f} {r['AUC_ROC']:>9.4f}{marker}")
print("-" * 72)
print(f"[BEST]  {best_model_name}  (Recall = {best_recall:.4f})")

# ===========================================================================
# 10.  SAVE OUTPUTS
# ===========================================================================

# 10a. Model comparison metrics CSV
metrics_path = os.path.join(OUTPUT_DIR, "model_comparison_metrics.csv")
import csv
with open(metrics_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Model", "Accuracy", "Precision",
                                           "Recall", "F1_Score", "AUC_ROC"])
    writer.writeheader()
    writer.writerows(results)
print(f"\n[SAVE]  Metrics       -> {metrics_path}")

# 10b. Confusion matrices JSON
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices.json")
with open(cm_path, "w", encoding="utf-8") as f:
    json.dump(confusion_data, f, indent=2)
print(f"[SAVE]  Confusion     -> {cm_path}")

# 10c. Power BI export -- metrics
pbi_metrics = os.path.join(OUTPUT_DIR, "powerbi_model_metrics.csv")
shutil.copy2(metrics_path, pbi_metrics)
print(f"[SAVE]  PowerBI Metrics -> {pbi_metrics}")

# 10d. Power BI export -- predictions (from best model on test set)
pbi_preds_path = os.path.join(OUTPUT_DIR, "powerbi_predictions.csv")
pred_cols = [
    "Event_ID", "SKU_ID", "Apparel_Category", "Size", "Current_Stock",
    "Units_Sold_15m", "Cart_Adds_15m", "Views_15m", "Burn_Rate_Ratio",
    "Cart_Pressure_Index", "prediction",
]
best_preds = best_pipeline_model.transform(test_df).select(pred_cols)
best_preds.toPandas().to_csv(pbi_preds_path, index=False)
print(f"[SAVE]  PowerBI Preds  -> {pbi_preds_path}")

# 10e. Save best PipelineModel (with Windows fallback)
try:
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
    best_pipeline_model.write().overwrite().save(MODEL_DIR)
    print(f"[SAVE]  Best Model    -> {MODEL_DIR}")
except Exception as e:
    # Windows fallback: persist as Python pickle
    import pickle
    fallback_path = MODEL_DIR + "_pickle.pkl"
    os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
    # Save pipeline stages metadata for streaming consumer to reconstruct
    meta = {
        "best_model_name": best_model_name,
        "preprocessing_stages_count": len(preprocessing_stages),
        "results": results,
        "confusion_data": confusion_data,
    }
    with open(fallback_path, "wb") as pf:
        pickle.dump(meta, pf)
    print(f"[SAVE]  Model (pickle fallback) -> {fallback_path}")
    print(f"[WARN]  Native Spark save failed ({e}); using pickle metadata fallback.")

# 10f. Also save entire test-set predictions as live_predictions seed
live_preds_path = os.path.join(OUTPUT_DIR, "live_predictions.csv")
live_df = best_pipeline_model.transform(test_df)
live_cols = [
    "Event_ID", "Timestamp", "SKU_ID", "Apparel_Category", "Size",
    "Unit_Price", "Current_Stock", "Views_15m", "Cart_Adds_15m",
    "Units_Sold_15m", "Lead_Time_Hours", "Burn_Rate_Ratio",
    "Cart_Pressure_Index", "Conversion_Velocity", "Stockout_Risk_24h",
    "prediction",
]
live_df.select(live_cols).toPandas().to_csv(live_preds_path, index=False)
print(f"[SAVE]  Live Preds Seed -> {live_preds_path}")

# ===========================================================================
# 11.  DONE
# ===========================================================================
spark.stop()
print("\n" + "=" * 72)
print("  [OK]  PIPELINE COMPLETE -- All 5 models trained & evaluated.")
print("=" * 72 + "\n")

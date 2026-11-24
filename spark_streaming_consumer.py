#!/usr/bin/env python3
"""
=============================================================================
 SPARK STREAMING CONSUMER -- Real-Time Stockout Risk Inference
 File  : spark_streaming_consumer.py
 Usage : python spark_streaming_consumer.py              # file mode (default)
         python spark_streaming_consumer.py --mode kafka  # Kafka source
=============================================================================
 - Reads live JSON events from stream_inbox/ (or Kafka topic).
 - Engineers domain features on-the-fly.
 - Loads the pre-trained best PipelineModel and runs model.transform().
 - Outputs live predictions to console AND appends to outputs/live_predictions.csv.
 - Windows-compatible: uses polling-based file reading when Spark Structured
   Streaming is unavailable (no HADOOP_HOME / winutils.exe).
=============================================================================
"""

import os
import sys
import argparse
import warnings
import time
import json
import glob

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# -- Java 17 compatibility for PySpark ----------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_JDK17_PATH = os.path.join(PROJECT_ROOT, "jdk17", "jdk-17.0.20.1+1")
if os.path.isdir(_JDK17_PATH):
    os.environ["JAVA_HOME"] = _JDK17_PATH
    os.environ["PATH"] = os.path.join(_JDK17_PATH, "bin") + os.pathsep + os.environ.get("PATH", "")

# -- Hadoop winutils for Windows -----------------------------------------------
_HADOOP_HOME = os.path.join(PROJECT_ROOT, "hadoop")
if os.path.isdir(_HADOOP_HOME):
    os.environ["HADOOP_HOME"] = _HADOOP_HOME
    os.environ["PATH"] = os.path.join(_HADOOP_HOME, "bin") + os.pathsep + os.environ.get("PATH", "")

warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType,
)
from pyspark.sql.functions import col, lit, when, from_json, current_timestamp
from pyspark.ml import PipelineModel

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STREAM_DIR = os.path.join(PROJECT_ROOT, "stream_inbox")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "best_stockout_model")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "outputs", "live_predictions.csv")
KAFKA_TOPIC = "apparel_transactions"
KAFKA_BROKER = "localhost:9092"

# --- JSON Event Schema -----------------------------------------------------
EVENT_SCHEMA = StructType([
    StructField("Event_ID",         StringType(),  True),
    StructField("Timestamp",        StringType(),  True),
    StructField("SKU_ID",           StringType(),  True),
    StructField("Apparel_Category", StringType(),  True),
    StructField("Size",             StringType(),  True),
    StructField("Unit_Price",       DoubleType(),  True),
    StructField("Discount_Pct",     DoubleType(),  True),
    StructField("Current_Stock",    IntegerType(), True),
    StructField("Views_15m",        IntegerType(), True),
    StructField("Cart_Adds_15m",    IntegerType(), True),
    StructField("Units_Sold_15m",   IntegerType(), True),
    StructField("Lead_Time_Hours",  IntegerType(), True),
])


def add_engineered_features(df):
    """Add domain-engineered features matching the training pipeline."""
    return (
        df
        .withColumn("Burn_Rate_Ratio",
                     col("Units_Sold_15m") / (col("Current_Stock") + lit(1.0)))
        .withColumn("Cart_Pressure_Index",
                     col("Cart_Adds_15m") / (col("Current_Stock") + lit(1.0)))
        .withColumn("Conversion_Velocity",
                     col("Units_Sold_15m") / (col("Views_15m") + lit(1.0)))
    )


def process_batch_with_model(batch_df, batch_id, model):
    """Process a batch using the pre-trained PipelineModel."""
    filled = batch_df.na.fill({"Discount_Pct": 0.0, "Views_15m": 0})
    featured = add_engineered_features(filled)
    predictions = model.transform(featured)

    out_cols = [
        "Event_ID", "Timestamp", "SKU_ID", "Apparel_Category", "Size",
        "Unit_Price", "Current_Stock", "Views_15m", "Cart_Adds_15m",
        "Units_Sold_15m", "Lead_Time_Hours", "Burn_Rate_Ratio",
        "Cart_Pressure_Index", "Conversion_Velocity", "prediction",
    ]
    result = predictions.select(out_cols)
    result = result.withColumn(
        "Alert",
        when(col("prediction") == 1.0, "[ALERT] CRITICAL RESTOCK ALERT")
        .otherwise("[OK] SAFE")
    )
    return result


def process_batch_rule_based(batch_df, batch_id):
    """Process a batch using rule-based prediction (fallback)."""
    filled = batch_df.na.fill({"Discount_Pct": 0.0, "Views_15m": 0})
    featured = add_engineered_features(filled)
    result = featured.withColumn(
        "prediction",
        when(col("Burn_Rate_Ratio") > 0.3, lit(1.0)).otherwise(lit(0.0))
    ).withColumn(
        "Alert",
        when(col("prediction") == 1.0, "[ALERT] CRITICAL RESTOCK ALERT")
        .otherwise("[OK] SAFE")
    )
    return result


def print_batch(result, batch_id, mode_label=""):
    """Print batch results to console and append to CSV."""
    rows = result.collect()
    high_risk_count = sum(1 for r in rows if r["prediction"] == 1.0)

    print(f"\n{'-' * 80}")
    print(f"  BATCH {batch_id} {mode_label} |  {len(rows)} events  |  "
          f"{high_risk_count} HIGH-RISK alerts")
    print(f"{'-' * 80}")
    for r in rows:
        risk = "[ALERT] HIGH" if r["prediction"] == 1.0 else "   [OK]  LOW "
        sku = r["SKU_ID"] if r["SKU_ID"] else "?"
        cat = r["Apparel_Category"] if r["Apparel_Category"] else "?"
        stock = r["Current_Stock"] if r["Current_Stock"] is not None else 0
        sold = r["Units_Sold_15m"] if r["Units_Sold_15m"] is not None else 0
        burn = r["Burn_Rate_Ratio"] if r["Burn_Rate_Ratio"] is not None else 0.0
        print(f"  {risk}  {sku:>12}  {cat:<20} "
              f"stock={stock:>3}  sold_15m={sold:>2}  "
              f"burn={burn:.3f}")

    # Append to CSV
    pdf = result.toPandas()
    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    pdf.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)


def try_structured_streaming(spark, model, args):
    """
    Attempt native Spark Structured Streaming.
    Returns True if it started successfully, False if it failed (Windows/no Hadoop).
    """
    try:
        if args.mode == "kafka":
            print(f"[STREAM] Kafka source: {KAFKA_BROKER} / {KAFKA_TOPIC}")
            raw_stream = (
                spark.readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", KAFKA_BROKER)
                .option("subscribe", KAFKA_TOPIC)
                .option("startingOffsets", "latest")
                .load()
                .selectExpr("CAST(value AS STRING) as json_str")
                .select(from_json(col("json_str"), EVENT_SCHEMA).alias("data"))
                .select("data.*")
            )
        else:
            print(f"[STREAM] File source: {STREAM_DIR}/")
            raw_stream = (
                spark.readStream
                .schema(EVENT_SCHEMA)
                .option("maxFilesPerTrigger", 5)
                .json(STREAM_DIR)
            )

        if model is not None:
            def _process(batch_df, batch_id):
                if batch_df.rdd.isEmpty():
                    return
                result = process_batch_with_model(batch_df, batch_id, model)
                print_batch(result, batch_id)
        else:
            def _process(batch_df, batch_id):
                if batch_df.rdd.isEmpty():
                    return
                result = process_batch_rule_based(batch_df, batch_id)
                print_batch(result, batch_id, "(rule-based)")

        print("[STREAM] Starting streaming query ... (Press Ctrl+C to stop)\n")
        query = (
            raw_stream.writeStream
            .foreachBatch(_process)
            .trigger(processingTime="5 seconds")
            .option("checkpointLocation",
                    os.path.join(PROJECT_ROOT, "_checkpoints", "streaming"))
            .start()
        )
        query.awaitTermination()
        return True

    except KeyboardInterrupt:
        print("\n[STREAM] Stopping ...")
        return True
    except Exception as e:
        print(f"[STREAM] [WARN]  Structured Streaming failed: {e}")
        print("[STREAM] Falling back to polling-based file consumer ...\n")
        return False


def polling_consumer(spark, model):
    """
    Polling-based consumer that works on Windows without Hadoop/winutils.
    Reads new JSON files from stream_inbox/, processes them, and appends
    results to the live predictions CSV.
    """
    print(f"[STREAM] Polling-based consumer watching: {STREAM_DIR}/")
    print("[STREAM] Press Ctrl+C to stop.\n")

    os.makedirs(STREAM_DIR, exist_ok=True)
    processed_files = set()
    batch_id = 0

    # Mark already-existing files as processed to only process NEW events
    existing = set(glob.glob(os.path.join(STREAM_DIR, "*.json")))
    processed_files = existing.copy()
    if existing:
        print(f"[STREAM] Skipped {len(existing)} pre-existing files. Waiting for new events ...\n")

    try:
        while True:
            # Find new JSON files
            all_files = set(glob.glob(os.path.join(STREAM_DIR, "*.json")))
            new_files = sorted(all_files - processed_files)

            if new_files:
                # Process in batches of up to 10 files
                for i in range(0, len(new_files), 10):
                    chunk = new_files[i:i + 10]
                    batch_id += 1

                    try:
                        # Read JSON files as batch DataFrame
                        batch_df = spark.read.schema(EVENT_SCHEMA).json(chunk)
                        row_count = batch_df.count()

                        if row_count == 0:
                            processed_files.update(chunk)
                            continue

                        # Process with model or rule-based
                        if model is not None:
                            result = process_batch_with_model(batch_df, batch_id, model)
                            print_batch(result, batch_id)
                        else:
                            result = process_batch_rule_based(batch_df, batch_id)
                            print_batch(result, batch_id, "(rule-based)")

                    except Exception as e:
                        print(f"[STREAM] [WARN]  Error processing batch {batch_id}: {e}")

                    processed_files.update(chunk)

            # Poll every 5 seconds
            time.sleep(5)

    except KeyboardInterrupt:
        print(f"\n[STREAM] Stopped. Processed {batch_id} batches total.")


def main():
    parser = argparse.ArgumentParser(description="Spark Streaming Consumer")
    parser.add_argument("--mode", choices=["file", "kafka"], default="file",
                        help="Source mode: 'file' reads stream_inbox/, 'kafka' reads from Kafka.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SPARK STRUCTURED STREAMING CONSUMER")
    print("=" * 60)

    # -- Spark Session --------------------------------------------------------
    spark = (
        SparkSession.builder
        .appName("Apparel_Streaming_Consumer")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "1g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # -- Load Pre-trained Model ----------------------------------------------
    print(f"[STREAM] Loading model from {MODEL_DIR} ...")
    model = None
    # Check if the Spark-native model directory actually has content
    if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
        try:
            model = PipelineModel.load(MODEL_DIR)
            print("[STREAM] [OK] Model loaded successfully.")
        except Exception as e:
            print(f"[STREAM] [WARN] Could not load Spark model ({e}).")
    else:
        print("[STREAM] [WARN] Model directory is empty (Windows/no-Hadoop fallback).")

    if model is None:
        print("[STREAM] [WARN] Running in pass-through mode (rule-based fallback).")

    # -- Create stream_inbox directory ---------------------------------------
    os.makedirs(STREAM_DIR, exist_ok=True)

    # -- Try Structured Streaming first, fall back to polling ----------------
    if args.mode == "kafka":
        # Kafka mode: must use Structured Streaming
        try_structured_streaming(spark, model, args)
    else:
        # File mode: try Structured Streaming, fall back to polling on Windows
        success = try_structured_streaming(spark, model, args)
        if not success:
            polling_consumer(spark, model)

    spark.stop()
    print("[STREAM] [OK] Stopped.")


if __name__ == "__main__":
    main()

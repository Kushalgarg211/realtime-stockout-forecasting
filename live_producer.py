#!/usr/bin/env python3
"""
=============================================================================
 LIVE EVENT PRODUCER -- Simulates Real-Time E-Commerce Clickstream Events
 File  : live_producer.py
 Usage :
   python live_producer.py                  # default: file mode
   python live_producer.py --mode file      # writes JSON to stream_inbox/
   python live_producer.py --mode kafka     # publishes to Kafka topic
=============================================================================
 Generates one JSON event every ~2 seconds.
 Periodically injects "Flash Sale / Viral Trend" spikes on specific SKUs.
=============================================================================
"""

import os
import sys
import json
import time
import random
import argparse
import datetime
import uuid

# --- Configuration ----------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STREAM_DIR = os.path.join(PROJECT_ROOT, "stream_inbox")
KAFKA_TOPIC = "apparel_transactions"
KAFKA_BROKER = "localhost:9092"
EVENT_INTERVAL_SEC = 2.0
SPIKE_PROBABILITY = 0.15  # 15 % chance of a "viral spike" event

CATEGORIES = {
    "Oversized_Hoodie": {"sku_prefix": "SKU_HD_", "sku_range": (101, 150),
                         "price_range": (1299.0, 2999.0)},
    "Graphic_Tee":      {"sku_prefix": "SKU_GT_", "sku_range": (201, 250),
                         "price_range": (499.0, 1299.0)},
    "Sweatshirt":       {"sku_prefix": "SKU_SW_", "sku_range": (301, 350),
                         "price_range": (999.0, 2499.0)},
    "Jogger":           {"sku_prefix": "SKU_JG_", "sku_range": (401, 450),
                         "price_range": (799.0, 1999.0)},
    "Footwear":         {"sku_prefix": "SKU_FW_", "sku_range": (501, 550),
                         "price_range": (599.0, 1799.0)},
}
SIZES = ["S", "M", "L", "XL", "XXL"]
LEAD_TIMES = [12, 24, 48, 72]


# --- Event Generator --------------------------------------------------------
def generate_event(event_num: int, spike: bool = False) -> dict:
    """Generate a single realistic JSON event."""
    cat_name = random.choice(list(CATEGORIES.keys()))

    # During spikes, favour hoodies and graphic tees
    if spike:
        cat_name = random.choice(["Oversized_Hoodie", "Graphic_Tee",
                                   "Oversized_Hoodie", "Graphic_Tee", cat_name])

    cat = CATEGORIES[cat_name]
    sku_num = random.randint(*cat["sku_range"])
    sku_id = f"{cat['sku_prefix']}{sku_num}"

    # Low stock during spikes
    if spike:
        current_stock = random.randint(5, 30)
        views_15m = random.randint(200, 600)
        cart_adds_15m = random.randint(40, 150)
        units_sold_15m = random.randint(20, 80)
        discount_pct = round(random.uniform(20.0, 60.0), 1)
    else:
        current_stock = random.randint(30, 500)
        views_15m = random.randint(0, 250)
        cart_adds_15m = random.randint(0, 50)
        units_sold_15m = random.randint(0, 25)
        discount_pct = round(random.choice([0.0] * 4 + [random.uniform(5.0, 40.0)]), 1)

    return {
        "Event_ID": f"LIVE_{uuid.uuid4().hex[:8].upper()}",
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "SKU_ID": sku_id,
        "Apparel_Category": cat_name,
        "Size": random.choice(SIZES),
        "Unit_Price": round(random.uniform(*cat["price_range"]), 2),
        "Discount_Pct": discount_pct,
        "Current_Stock": current_stock,
        "Views_15m": views_15m,
        "Cart_Adds_15m": cart_adds_15m,
        "Units_Sold_15m": units_sold_15m,
        "Lead_Time_Hours": random.choice(LEAD_TIMES),
    }


# --- File Mode Writer -------------------------------------------------------
def run_file_mode():
    """Write micro-batch JSON files into stream_inbox/ for Spark readStream."""
    os.makedirs(STREAM_DIR, exist_ok=True)
    print(f"[PRODUCER-FILE] Writing events to {STREAM_DIR}/")
    print(f"[PRODUCER-FILE] Press Ctrl+C to stop.\n")

    event_num = 0
    try:
        while True:
            spike = random.random() < SPIKE_PROBABILITY
            event = generate_event(event_num, spike=spike)
            event_num += 1

            filename = f"event_{int(time.time() * 1000)}_{event_num}.json"
            filepath = os.path.join(STREAM_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(event, f)

            tag = "[SPIKE] SPIKE" if spike else "  normal"
            print(f"  [{tag}]  {event['SKU_ID']:>12}  stock={event['Current_Stock']:>3}  "
                  f"sold_15m={event['Units_Sold_15m']:>2}  -> {filename}")

            time.sleep(EVENT_INTERVAL_SEC)
    except KeyboardInterrupt:
        print(f"\n[PRODUCER-FILE] Stopped after {event_num} events.")


# --- Kafka Mode Publisher ----------------------------------------------------
def run_kafka_mode():
    """Publish JSON events to the Kafka topic."""
    try:
        from kafka import KafkaProducer
    except ImportError:
        print("[ERROR] kafka-python not installed.  pip install kafka-python")
        sys.exit(1)

    print(f"[PRODUCER-KAFKA] Connecting to {KAFKA_BROKER} …")
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3,
        )
    except Exception as e:
        print(f"[ERROR] Cannot connect to Kafka broker: {e}")
        print("[HINT]  Falling back to file mode …\n")
        run_file_mode()
        return

    print(f"[PRODUCER-KAFKA] Publishing to topic '{KAFKA_TOPIC}'")
    print(f"[PRODUCER-KAFKA] Press Ctrl+C to stop.\n")

    event_num = 0
    try:
        while True:
            spike = random.random() < SPIKE_PROBABILITY
            event = generate_event(event_num, spike=spike)
            event_num += 1

            producer.send(KAFKA_TOPIC, value=event)
            producer.flush()

            tag = "[SPIKE] SPIKE" if spike else "  normal"
            print(f"  [{tag}]  {event['SKU_ID']:>12}  stock={event['Current_Stock']:>3}  "
                  f"sold_15m={event['Units_Sold_15m']:>2}")

            time.sleep(EVENT_INTERVAL_SEC)
    except KeyboardInterrupt:
        print(f"\n[PRODUCER-KAFKA] Stopped after {event_num} events.")
        producer.close()


# --- CLI Entry Point --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Live E-Commerce Event Producer (File or Kafka mode)")
    parser.add_argument("--mode", choices=["file", "kafka"], default="file",
                        help="Output mode: 'file' (default) writes JSON to stream_inbox/, "
                             "'kafka' publishes to Kafka topic.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LIVE EVENT PRODUCER -- E-Commerce Apparel Stream")
    print("=" * 60)

    if args.mode == "kafka":
        run_kafka_mode()
    else:
        run_file_mode()


if __name__ == "__main__":
    main()

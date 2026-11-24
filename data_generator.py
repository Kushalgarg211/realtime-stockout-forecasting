#!/usr/bin/env python3
"""
=============================================================================
 SYNTHETIC DATA GENERATOR — Real-Time Trend Forecasting & Inventory
 Optimization for E-Commerce Apparel
 Generates 50,000 realistic historical sales rows with correlated features,
 ~2% missing values, and ~2% duplicate rows.
=============================================================================
"""

import os
import csv
import random
import datetime
import math

# ─── Configuration ──────────────────────────────────────────────────────────
NUM_ROWS = 50_000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "historical_apparel_sales.csv")
MISSING_VALUE_RATE = 0.02   # 2 % cells replaced with blanks
DUPLICATE_RATE = 0.02       # 2 % rows duplicated at the end

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ─── Domain Constants ───────────────────────────────────────────────────────
CATEGORIES = {
    "Oversized_Hoodie": {"sku_prefix": "SKU_HD_", "sku_range": (101, 150),
                         "price_range": (1299.0, 2999.0), "trend_multiplier": 1.4},
    "Graphic_Tee":      {"sku_prefix": "SKU_GT_", "sku_range": (201, 250),
                         "price_range": (499.0, 1299.0),  "trend_multiplier": 1.3},
    "Sweatshirt":       {"sku_prefix": "SKU_SW_", "sku_range": (301, 350),
                         "price_range": (999.0, 2499.0),  "trend_multiplier": 1.1},
    "Jogger":           {"sku_prefix": "SKU_JG_", "sku_range": (401, 450),
                         "price_range": (799.0, 1999.0),  "trend_multiplier": 1.0},
    "Footwear":         {"sku_prefix": "SKU_FW_", "sku_range": (501, 550),
                         "price_range": (599.0, 1799.0),  "trend_multiplier": 0.9},
}
SIZES = ["S", "M", "L", "XL", "XXL"]
LEAD_TIMES = [12, 24, 48, 72]

COLUMNS = [
    "Event_ID", "Timestamp", "SKU_ID", "Apparel_Category", "Size",
    "Unit_Price", "Discount_Pct", "Current_Stock", "Views_15m",
    "Cart_Adds_15m", "Units_Sold_15m", "Lead_Time_Hours", "Stockout_Risk_24h",
]


# ─── Helper Functions ────────────────────────────────────────────────────────
def _random_timestamp(base: datetime.datetime, span_days: int = 90) -> str:
    """Return a random ISO-8601 timestamp within *span_days* of *base*."""
    offset = random.randint(0, span_days * 24 * 3600)
    ts = base + datetime.timedelta(seconds=offset)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _compute_stockout_label(current_stock: int, units_sold_15m: int,
                            cart_adds_15m: int, lead_time_hours: int,
                            discount_pct: float, views_15m: int) -> int:
    """
    Realistic heuristic for ground-truth stockout risk in the next 24 hours.
    High risk when demand velocity is high relative to remaining stock and
    replenishment lead time is long.
    """
    # Project 24-hour demand from the 15-minute window (96 windows in 24 h)
    projected_demand = (units_sold_15m * 96) * 0.35  # damped extrapolation
    projected_cart_demand = (cart_adds_15m * 96) * 0.10  # portion of carts convert
    total_demand_est = projected_demand + projected_cart_demand

    # If stock can cover projected demand, likely safe
    coverage_ratio = current_stock / (total_demand_est + 1.0)

    # Lead time factor: longer lead time → riskier
    lead_factor = lead_time_hours / 24.0

    # Risk score (0–1 range roughly)
    risk_score = (1.0 - min(coverage_ratio, 1.0)) * 0.55 \
                 + (lead_factor / 3.0) * 0.15 \
                 + (discount_pct / 60.0) * 0.10 \
                 + min(views_15m / 600.0, 1.0) * 0.10 \
                 + min(units_sold_15m / 80.0, 1.0) * 0.10

    # Add realistic noise (±0.08)
    risk_score += random.gauss(0, 0.08)

    return 1 if risk_score >= 0.58 else 0


def generate_row(event_id: int, base_ts: datetime.datetime) -> dict:
    """Generate a single realistic row."""
    cat_name = random.choice(list(CATEGORIES.keys()))
    cat = CATEGORIES[cat_name]
    sku_num = random.randint(*cat["sku_range"])
    sku_id = f"{cat['sku_prefix']}{sku_num}"

    size = random.choice(SIZES)
    unit_price = round(random.uniform(*cat["price_range"]), 2)
    discount_pct = round(random.choice([0.0] * 4 + [random.uniform(5.0, 60.0)]), 1)

    # Stock tends to cluster at lower values for trending items
    if random.random() < 0.35 * cat["trend_multiplier"]:
        current_stock = random.randint(5, 60)
    else:
        current_stock = random.randint(30, 500)

    # Views correlated with trending multiplier and discount
    base_views = int(random.triangular(0, 600, 80) * cat["trend_multiplier"])
    if discount_pct > 30:
        base_views = int(base_views * 1.6)
    views_15m = min(max(base_views, 0), 600)

    # Cart adds loosely correlated with views
    cart_rate = random.uniform(0.05, 0.35)
    cart_adds_15m = min(int(views_15m * cart_rate), 150)

    # Units sold correlated with cart adds
    sold_rate = random.uniform(0.15, 0.65)
    units_sold_15m = min(int(cart_adds_15m * sold_rate), 80)

    lead_time = random.choice(LEAD_TIMES)

    label = _compute_stockout_label(
        current_stock, units_sold_15m, cart_adds_15m,
        lead_time, discount_pct, views_15m,
    )

    return {
        "Event_ID": f"EVT_{100001 + event_id}",
        "Timestamp": _random_timestamp(base_ts),
        "SKU_ID": sku_id,
        "Apparel_Category": cat_name,
        "Size": size,
        "Unit_Price": unit_price,
        "Discount_Pct": discount_pct,
        "Current_Stock": current_stock,
        "Views_15m": views_15m,
        "Cart_Adds_15m": cart_adds_15m,
        "Units_Sold_15m": units_sold_15m,
        "Lead_Time_Hours": lead_time,
        "Stockout_Risk_24h": label,
    }


# ─── Main Generator ─────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_ts = datetime.datetime(2026, 7, 1, 0, 0, 0)

    print(f"[DATA-GEN] Generating {NUM_ROWS:,} rows …")
    rows = [generate_row(i, base_ts) for i in range(NUM_ROWS)]

    # ── Inject ~2 % missing values in Discount_Pct and Views_15m ────────────
    missing_count = 0
    for row in rows:
        if random.random() < MISSING_VALUE_RATE:
            row["Discount_Pct"] = ""
            missing_count += 1
        if random.random() < MISSING_VALUE_RATE:
            row["Views_15m"] = ""
            missing_count += 1
    print(f"[DATA-GEN] Injected {missing_count} missing-value cells.")

    # ── Inject ~2 % duplicate rows ──────────────────────────────────────────
    num_dupes = int(NUM_ROWS * DUPLICATE_RATE)
    dupes = random.choices(rows, k=num_dupes)
    rows.extend(dupes)
    random.shuffle(rows)
    print(f"[DATA-GEN] Added {num_dupes} duplicate rows -> total {len(rows):,} rows.")

    # ── Write CSV ───────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"[DATA-GEN] [OK] Saved {OUTPUT_FILE}  ({file_size_mb:.2f} MB, {len(rows):,} rows)")

    # ── Quick label distribution ────────────────────────────────────────────
    ones = sum(1 for r in rows if r.get("Stockout_Risk_24h") == 1)
    zeros = len(rows) - ones
    print(f"[DATA-GEN] Label distribution -> 0 (Low Risk): {zeros:,}  |  1 (High Risk): {ones:,}")
    print(f"[DATA-GEN]                       Class-1 ratio: {ones / len(rows) * 100:.1f} %")


if __name__ == "__main__":
    main()

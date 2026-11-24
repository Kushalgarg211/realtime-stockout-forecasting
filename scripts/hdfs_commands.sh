#!/usr/bin/env bash
# =============================================================================
#  HDFS Upload Commands — Historical Apparel Sales Dataset
#  File  : scripts/hdfs_commands.sh
#  Usage : bash scripts/hdfs_commands.sh
#  Prereq: Hadoop HDFS daemons running (NameNode + DataNode)
# =============================================================================

set -euo pipefail

HDFS_BASE="/user/hive/warehouse/ecommerce_analytics.db/apparel_historical_sales"
LOCAL_CSV="data/historical_apparel_sales.csv"
CATEGORIES=("Oversized_Hoodie" "Graphic_Tee" "Sweatshirt" "Jogger" "Footwear")

echo "═══════════════════════════════════════════════════════"
echo "  HDFS Upload — Apparel Historical Sales Dataset"
echo "═══════════════════════════════════════════════════════"

# 1. Verify that the local CSV exists
if [ ! -f "$LOCAL_CSV" ]; then
    echo "[ERROR] Local file not found: $LOCAL_CSV"
    echo "        Run 'python data_generator.py' first."
    exit 1
fi

# 2. Create HDFS directories for each Hive partition
echo "[HDFS] Creating partition directories …"
for CAT in "${CATEGORIES[@]}"; do
    hdfs dfs -mkdir -p "${HDFS_BASE}/Apparel_Category=${CAT}"
    echo "       ✔  ${HDFS_BASE}/Apparel_Category=${CAT}"
done

# 3. Split the CSV by category and upload each partition
echo "[HDFS] Splitting CSV and uploading partitions …"
HEADER=$(head -1 "$LOCAL_CSV")

for CAT in "${CATEGORIES[@]}"; do
    TMP_FILE="/tmp/apparel_${CAT}.csv"
    # Extract header + matching rows
    echo "$HEADER" > "$TMP_FILE"
    grep ",${CAT}," "$LOCAL_CSV" >> "$TMP_FILE" || true

    ROW_COUNT=$(( $(wc -l < "$TMP_FILE") - 1 ))
    echo "       ${CAT}: ${ROW_COUNT} rows"

    hdfs dfs -put -f "$TMP_FILE" "${HDFS_BASE}/Apparel_Category=${CAT}/data.csv"
    rm -f "$TMP_FILE"
done

# 4. Verify uploads
echo ""
echo "[HDFS] Verification — listing partition directories:"
hdfs dfs -ls -R "$HDFS_BASE" | head -30

# 5. Quick byte-count sanity check
TOTAL_BYTES=$(hdfs dfs -du -s "$HDFS_BASE" | awk '{print $1}')
TOTAL_MB=$(echo "scale=2; $TOTAL_BYTES / 1048576" | bc)
echo ""
echo "[HDFS] ✅  Total uploaded: ${TOTAL_MB} MB"
echo "[HDFS] Done.  Run 'hive -f scripts/hive_setup.hql' to create the Hive table."

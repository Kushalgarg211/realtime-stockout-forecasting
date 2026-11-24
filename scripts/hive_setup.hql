-- =============================================================================
--  HIVE DDL — External Partitioned Table for Historical Apparel Sales
--  File: scripts/hive_setup.hql
--  Usage: hive -f scripts/hive_setup.hql
-- =============================================================================

-- 1. Create the analytics database (idempotent)
CREATE DATABASE IF NOT EXISTS ecommerce_analytics
COMMENT 'Database for real-time apparel inventory forecasting project';

USE ecommerce_analytics;

-- 2. Drop & recreate the external table (clean re-runs)
DROP TABLE IF EXISTS apparel_historical_sales;

-- 3. External table mapped to HDFS CSV location
--    Partitioned by Apparel_Category for efficient category-level queries.
CREATE EXTERNAL TABLE apparel_historical_sales (
    Event_ID          STRING   COMMENT 'Unique event identifier (EVT_100001 …)',
    Timestamp         TIMESTAMP COMMENT 'Event timestamp ISO-8601',
    SKU_ID            STRING   COMMENT 'Stock-keeping unit ID',
    Size              STRING   COMMENT 'Apparel size: S, M, L, XL, XXL',
    Unit_Price        DOUBLE   COMMENT 'Price in INR',
    Discount_Pct      DOUBLE   COMMENT 'Discount percentage (0–60)',
    Current_Stock     INT      COMMENT 'Warehouse stock at event time',
    Views_15m         INT      COMMENT 'Page views in trailing 15 min window',
    Cart_Adds_15m     INT      COMMENT 'Cart additions in trailing 15 min window',
    Units_Sold_15m    INT      COMMENT 'Checkout units in trailing 15 min window',
    Lead_Time_Hours   INT      COMMENT 'Supplier restock lead time (12/24/48/72 h)',
    Stockout_Risk_24h INT      COMMENT 'Label: 1 = High stockout risk, 0 = Low'
)
PARTITIONED BY (Apparel_Category STRING)
ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '/user/hive/warehouse/ecommerce_analytics.db/apparel_historical_sales'
TBLPROPERTIES ('skip.header.line.count'='1');

-- 4. Manually add partitions after HDFS upload
ALTER TABLE apparel_historical_sales ADD IF NOT EXISTS
    PARTITION (Apparel_Category = 'Oversized_Hoodie')
    LOCATION '/user/hive/warehouse/ecommerce_analytics.db/apparel_historical_sales/Apparel_Category=Oversized_Hoodie';

ALTER TABLE apparel_historical_sales ADD IF NOT EXISTS
    PARTITION (Apparel_Category = 'Graphic_Tee')
    LOCATION '/user/hive/warehouse/ecommerce_analytics.db/apparel_historical_sales/Apparel_Category=Graphic_Tee';

ALTER TABLE apparel_historical_sales ADD IF NOT EXISTS
    PARTITION (Apparel_Category = 'Sweatshirt')
    LOCATION '/user/hive/warehouse/ecommerce_analytics.db/apparel_historical_sales/Apparel_Category=Sweatshirt';

ALTER TABLE apparel_historical_sales ADD IF NOT EXISTS
    PARTITION (Apparel_Category = 'Jogger')
    LOCATION '/user/hive/warehouse/ecommerce_analytics.db/apparel_historical_sales/Apparel_Category=Jogger';

ALTER TABLE apparel_historical_sales ADD IF NOT EXISTS
    PARTITION (Apparel_Category = 'Footwear')
    LOCATION '/user/hive/warehouse/ecommerce_analytics.db/apparel_historical_sales/Apparel_Category=Footwear';


-- ═══════════════════════════════════════════════════════════════════════════
--  SAMPLE ANALYTICAL QUERIES
-- ═══════════════════════════════════════════════════════════════════════════

-- Q1: Overall stockout risk distribution
SELECT
    Stockout_Risk_24h,
    COUNT(*)           AS event_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
FROM apparel_historical_sales
GROUP BY Stockout_Risk_24h;

-- Q2: Category-wise average sales velocity & stockout rate
SELECT
    Apparel_Category,
    ROUND(AVG(Units_Sold_15m), 2)       AS avg_velocity,
    ROUND(AVG(Current_Stock), 0)         AS avg_stock,
    ROUND(AVG(Stockout_Risk_24h) * 100, 2) AS stockout_pct
FROM apparel_historical_sales
GROUP BY Apparel_Category
ORDER BY stockout_pct DESC;

-- Q3: Top-10 SKUs by cumulative 15-min sales velocity
SELECT
    SKU_ID,
    Apparel_Category,
    SUM(Units_Sold_15m) AS total_velocity,
    AVG(Current_Stock)  AS avg_stock
FROM apparel_historical_sales
GROUP BY SKU_ID, Apparel_Category
ORDER BY total_velocity DESC
LIMIT 10;

-- Q4: Hourly trend — stockout events over time
SELECT
    HOUR(Timestamp)   AS event_hour,
    SUM(Stockout_Risk_24h) AS high_risk_count
FROM apparel_historical_sales
GROUP BY HOUR(Timestamp)
ORDER BY event_hour;

-- Q5: Discount impact on stockout risk
SELECT
    CASE
        WHEN Discount_Pct = 0            THEN 'No Discount'
        WHEN Discount_Pct BETWEEN 1 AND 20  THEN 'Low (1-20%)'
        WHEN Discount_Pct BETWEEN 21 AND 40 THEN 'Medium (21-40%)'
        ELSE 'High (41-60%)'
    END AS discount_band,
    COUNT(*)                              AS events,
    ROUND(AVG(Stockout_Risk_24h) * 100, 2) AS stockout_pct
FROM apparel_historical_sales
GROUP BY
    CASE
        WHEN Discount_Pct = 0            THEN 'No Discount'
        WHEN Discount_Pct BETWEEN 1 AND 20  THEN 'Low (1-20%)'
        WHEN Discount_Pct BETWEEN 21 AND 40 THEN 'Medium (21-40%)'
        ELSE 'High (41-60%)'
    END
ORDER BY stockout_pct DESC;

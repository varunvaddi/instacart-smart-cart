# Databricks notebook source
# Cell 1: Silver Layer Setup & Environment Check

print("=" * 70)
print("🥈 SILVER LAYER TRANSFORMATION - DAY 2")
print("=" * 70)

# Test cluster
print(f"\n📊 CLUSTER INFO:")
print(f"   Spark Version: {spark.version}")

# Check Bronze database exists
print(f"\n🔍 CHECKING BRONZE DATABASE:")
bronze_tables = spark.sql("SHOW TABLES IN bronze_db").collect()
print(f"   ✅ Bronze database found with {len(bronze_tables)} tables:")
for table in bronze_tables:
    count = spark.table(f"bronze_db.{table.tableName}").count()
    print(f"      - {table.tableName:20s}: {count:>12,} rows")

# Quick data quality check
print(f"\n🧪 BRONZE DATA QUALITY:")
orders = spark.table("bronze_db.orders")
order_products = spark.table("bronze_db.order_products")

print(f"   Orders: {orders.count():,}")
print(f"   Order Products: {order_products.count():,}")
print(f"   Unique users: {orders.select('user_id').distinct().count():,}")

print("\n" + "=" * 70)
print("✅ READY TO BUILD SILVER LAYER!")
print("=" * 70)

# COMMAND ----------

# Cell 2: Create Silver Database

print("=" * 70)
print("📁 CREATING SILVER DATABASE")
print("=" * 70)

# Create database
spark.sql("CREATE DATABASE IF NOT EXISTS silver_db")
print("\n✅ Database 'silver_db' created")

# Verify
databases = spark.sql("SHOW DATABASES").collect()
db_names = [row.databaseName for row in databases]

print(f"\n📊 Available Databases:")
for db in db_names:
    print(f"   - {db}")

# Set default database
spark.sql("USE silver_db")
print(f"\n✅ Using database: silver_db")

print("\n🎯 SILVER LAYER TABLES WE'LL CREATE:")
print("   1. orders_cleaned          - Validated orders with quality checks")
print("   2. products_enriched       - Products + aisles + departments")
print("   3. user_order_history      - Orders per user with metrics")
print("   4. order_products_enriched - Order items + product details")

print("\n" + "=" * 70)

# COMMAND ----------

# Cell 3: Clean Orders Table

print("=" * 70)
print("🧹 CLEANING ORDERS TABLE")
print("=" * 70)

from pyspark.sql.functions import col, when, current_timestamp

# Load Bronze orders
print("\n📥 Loading Bronze orders...")
orders_bronze = spark.table("bronze_db.orders")
initial_count = orders_bronze.count()
print(f"   Initial rows: {initial_count:,}")

# Step 1: Remove duplicates
print("\n1️⃣  REMOVING DUPLICATES:")
orders_deduped = orders_bronze.dropDuplicates(["order_id"])
dedup_count = orders_deduped.count()
duplicates_removed = initial_count - dedup_count
print(f"   Removed: {duplicates_removed:,} duplicates")
print(f"   Remaining: {dedup_count:,} rows")

# Step 2: Validate required fields
print("\n2️⃣  VALIDATING REQUIRED FIELDS:")
orders_valid = orders_deduped.filter(
    col("order_id").isNotNull() &
    col("user_id").isNotNull() &
    col("order_number").isNotNull()
)
valid_count = orders_valid.count()
invalid_removed = dedup_count - valid_count
print(f"   Removed: {invalid_removed:,} rows with null IDs")
print(f"   Remaining: {valid_count:,} rows")

# Step 3: Validate value ranges
print("\n3️⃣  VALIDATING VALUE RANGES:")
orders_clean = orders_valid.filter(
    col("order_dow").between(0, 6) &
    col("order_hour_of_day").between(0, 23)
)
clean_count = orders_clean.count()
range_invalid = valid_count - clean_count
print(f"   Removed: {range_invalid:,} rows with invalid ranges")
print(f"   Remaining: {clean_count:,} rows")

# Step 4: Handle nulls in days_since_prior_order
print("\n4️⃣  HANDLING NULLS:")
orders_clean = orders_clean.withColumn(
    "days_since_prior_order",
    when(col("days_since_prior_order").isNull(), 0.0)
    .otherwise(col("days_since_prior_order"))
)
print(f"   ✅ Filled null days_since_prior_order with 0.0")

# Add Silver metadata
print("\n➕ ADDING SILVER METADATA:")
orders_silver = orders_clean \
    .withColumn("silver_processing_time", current_timestamp()) \
    .withColumn("quality_status", when(col("order_id").isNotNull(), "CLEAN").otherwise("INVALID"))

print(f"   ✅ Added: silver_processing_time, quality_status")

# Summary
print("\n📊 CLEANING SUMMARY:")
print(f"   Initial:     {initial_count:>12,} rows")
print(f"   Duplicates:  {duplicates_removed:>12,} removed")
print(f"   Invalid:     {invalid_removed + range_invalid:>12,} removed")
print(f"   Final:       {clean_count:>12,} rows")
print(f"   Clean rate:  {(clean_count / initial_count * 100):>11.2f}%")

# Show sample
print("\n📋 CLEANED DATA SAMPLE:")
orders_silver.show(5, truncate=False)

print("\n" + "=" * 70)
print("✅ ORDERS CLEANED!")
print("=" * 70)

# COMMAND ----------

# Cell 4: Save Orders to Silver

print("=" * 70)
print("💾 SAVING CLEANED ORDERS TO SILVER")
print("=" * 70)

# Write to Delta
print("\n💾 Writing to Delta Lake...")
orders_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("silver_db.orders_cleaned")

print("   ✅ Saved as silver_db.orders_cleaned")

# Verify
result = spark.table("silver_db.orders_cleaned")
print(f"\n✅ Verification:")
print(f"   Rows: {result.count():,}")
print(f"   Columns: {len(result.columns)}")

# Schema
print(f"\n📋 Schema:")
result.printSchema()

print("\n" + "=" * 70)
print("✅ ORDERS SAVED TO SILVER!")
print("=" * 70)
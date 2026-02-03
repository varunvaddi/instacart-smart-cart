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

# COMMAND ----------

# Cell 5: Enrich Products with Categories

print("=" * 70)
print("📦 ENRICHING PRODUCTS TABLE")
print("=" * 70)

from pyspark.sql.functions import col, current_timestamp, upper, trim

# Load Bronze tables
print("\n📥 Loading Bronze tables...")
products = spark.table("bronze_db.products")
aisles = spark.table("bronze_db.aisles")
departments = spark.table("bronze_db.departments")

print(f"   Products:    {products.count():,} rows")
print(f"   Aisles:      {aisles.count():,} rows")
print(f"   Departments: {departments.count():,} rows")

# Step 1: Clean product names
print("\n1️⃣  CLEANING PRODUCT NAMES:")
products_clean = products \
    .withColumn("product_name_clean", trim(col("product_name"))) \
    .dropDuplicates(["product_id"])

print(f"   ✅ Trimmed whitespace, removed duplicates")
print(f"   Rows: {products_clean.count():,}")

# Step 2: Join with aisles
print("\n2️⃣  JOINING WITH AISLES:")
products_with_aisles = products_clean.join(
    aisles,
    products_clean.aisle_id == aisles.aisle_id,
    "left"
).select(
    products_clean["*"],
    aisles["aisle"].alias("aisle_name")
)

print(f"   ✅ Joined with aisles")
print(f"   Rows: {products_with_aisles.count():,}")

# Step 3: Join with departments
print("\n3️⃣  JOINING WITH DEPARTMENTS:")
products_enriched = products_with_aisles.join(
    departments,
    products_with_aisles.department_id == departments.department_id,
    "left"
).select(
    products_with_aisles["product_id"],
    products_with_aisles["product_name"],
    products_with_aisles["product_name_clean"],
    products_with_aisles["aisle_id"],
    products_with_aisles["aisle_name"],
    products_with_aisles["department_id"],
    departments["department"].alias("department_name")
)

print(f"   ✅ Joined with departments")
print(f"   Rows: {products_enriched.count():,}")

# Step 4: Add metadata
print("\n4️⃣  ADDING METADATA:")
products_silver = products_enriched \
    .withColumn("silver_processing_time", current_timestamp())

print(f"   ✅ Added silver_processing_time")

# Show sample
print("\n📊 ENRICHED PRODUCTS SAMPLE:")
products_silver.show(10, truncate=False)

# Validation
print("\n🧪 VALIDATION:")
null_aisles = products_silver.filter(col("aisle_name").isNull()).count()
null_depts = products_silver.filter(col("department_name").isNull()).count()

print(f"   Null aisles:      {null_aisles:,}")
print(f"   Null departments: {null_depts:,}")

if null_aisles == 0 and null_depts == 0:
    print(f"   ✅ All products have categories!")

print("\n" + "=" * 70)
print("✅ PRODUCTS ENRICHED!")
print("=" * 70)

# COMMAND ----------

# Cell 6: Save Products Enriched to Silver

print("=" * 70)
print("💾 SAVING PRODUCTS ENRICHED TO SILVER")
print("=" * 70)

# Write to Delta
print("\n💾 Writing to Delta Lake...")
products_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("silver_db.products_enriched")

print("   ✅ Saved as silver_db.products_enriched")

# Verify
result = spark.table("silver_db.products_enriched")
print(f"\n✅ Verification:")
print(f"   Rows: {result.count():,}")
print(f"   Columns: {len(result.columns)}")

# Show schema
print(f"\n📋 Schema:")
result.printSchema()

# Sample by department
print(f"\n📊 Products by Department (Top 10):")
dept_counts = result.groupBy("department_name").count() \
    .orderBy(col("count").desc())
dept_counts.show(10, truncate=False)

print("\n" + "=" * 70)
print("✅ PRODUCTS ENRICHED SAVED!")
print("=" * 70)

# COMMAND ----------

# Cell 7: Create User Order History

print("=" * 70)
print("👥 CREATING USER ORDER HISTORY")
print("=" * 70)

from pyspark.sql.functions import (
    count, max, min, avg, sum, 
    datediff, current_timestamp, lit
)

# Load cleaned orders
print("\n📥 Loading cleaned orders...")
orders = spark.table("silver_db.orders_cleaned")
print(f"   Orders: {orders.count():,}")

# Aggregate by user
print("\n📊 AGGREGATING BY USER:")
user_metrics = orders.groupBy("user_id").agg(
    count("order_id").alias("total_orders"),
    max("order_number").alias("max_order_number"),
    min("order_dow").alias("first_order_dow"),
    max("order_dow").alias("last_order_dow"),
    avg("order_hour_of_day").alias("avg_order_hour"),
    max("days_since_prior_order").alias("max_days_between_orders"),
    avg("days_since_prior_order").alias("avg_days_between_orders")
)

user_count = user_metrics.count()
print(f"   ✅ Aggregated {user_count:,} users")

# Add derived features
print("\n➕ ADDING DERIVED FEATURES:")

# Calculate user activity level
user_enriched = user_metrics.withColumn(
    "user_activity_level",
    when(col("total_orders") >= 20, "VERY_ACTIVE")
    .when(col("total_orders") >= 10, "ACTIVE")
    .when(col("total_orders") >= 5, "MODERATE")
    .otherwise("LOW")
)

# Add ordering preference (morning/afternoon/evening)
user_enriched = user_enriched.withColumn(
    "ordering_preference",
    when(col("avg_order_hour") < 10, "MORNING")
    .when(col("avg_order_hour") < 17, "AFTERNOON")
    .otherwise("EVENING")
)

# Add metadata
user_enriched = user_enriched.withColumn(
    "silver_processing_time", 
    current_timestamp()
)

print(f"   ✅ Added: user_activity_level, ordering_preference, metadata")

# Show sample
print("\n📊 USER METRICS SAMPLE:")
user_enriched.orderBy(col("total_orders").desc()).show(10, truncate=False)

# Distribution
print("\n📈 ACTIVITY LEVEL DISTRIBUTION:")
user_enriched.groupBy("user_activity_level").count() \
    .orderBy(col("count").desc()).show()

print("\n📈 ORDERING PREFERENCE DISTRIBUTION:")
user_enriched.groupBy("ordering_preference").count() \
    .orderBy(col("count").desc()).show()

print("\n" + "=" * 70)
print("✅ USER ORDER HISTORY CREATED!")
print("=" * 70)

# COMMAND ----------

# Cell 8: Save User Order History to Silver

print("=" * 70)
print("💾 SAVING USER ORDER HISTORY TO SILVER")
print("=" * 70)

# Write to Delta
print("\n💾 Writing to Delta Lake...")
user_enriched.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("silver_db.user_order_history")

print("   ✅ Saved as silver_db.user_order_history")

# Verify
result = spark.table("silver_db.user_order_history")
print(f"\n✅ Verification:")
print(f"   Rows: {result.count():,}")
print(f"   Columns: {len(result.columns)}")

# Schema
print(f"\n📋 Schema:")
result.printSchema()

print("\n" + "=" * 70)
print("✅ USER ORDER HISTORY SAVED!")
print("=" * 70)
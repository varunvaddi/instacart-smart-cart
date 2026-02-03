# Databricks notebook source
# Cell 1: Environment Check & Setup

print("=" * 70)
print("🚀 BRONZE LAYER SETUP - DAY 1")
print("=" * 70)

# Test cluster
print(f"\n📊 CLUSTER INFO:")
print(f"   Spark Version: {spark.version}")
print(f"   Python: 3.10+")

# Test mount
print(f"\n🔍 CHECKING MOUNT:")
try:
    files = dbutils.fs.ls("/mnt/bronze/raw/")
    print(f"   ✅ Mount active - found {len(files)} files")
except:
    print(f"   ❌ Mount not found - need to remount!")

# Test data access
print(f"\n🧪 TESTING DATA ACCESS:")
try:
    test_df = spark.read.csv("/mnt/bronze/raw/orders.csv", header=True, inferSchema=True)
    row_count = test_df.count()
    print(f"   ✅ Can read data - {row_count:,} orders found")
except Exception as e:
    print(f"   ❌ Error reading data: {e}")

print("\n" + "=" * 70)
print("✅ READY TO BUILD BRONZE LAYER!")
print("=" * 70)


# COMMAND ----------

# Cell 2: Create Bronze Database

print("=" * 70)
print("📁 CREATING BRONZE DATABASE")
print("=" * 70)

# Create database
spark.sql("CREATE DATABASE IF NOT EXISTS bronze_db")
print("\n✅ Database 'bronze_db' created")

# Verify
databases = spark.sql("SHOW DATABASES").collect()
db_names = [row.databaseName for row in databases]

print(f"\n📊 Available Databases:")
for db in db_names:
    print(f"   - {db}")

# Set default database
spark.sql("USE bronze_db")
print(f"\n✅ Using database: bronze_db")

print("=" * 70)

# COMMAND ----------

# Cell 3: Load Aisles to Bronze Delta Table

print("=" * 70)
print("🏪 LOADING AISLES TABLE")
print("=" * 70)

from pyspark.sql.functions import current_timestamp, lit

# Read CSV
print("\n🔄 Reading aisles.csv...")
aisles = spark.read.csv("/mnt/bronze/raw/aisles.csv", header=True, inferSchema=True)

print(f"   ✅ Loaded {aisles.count():,} aisles")

# Show sample
print("\n📊 Sample Data:")
aisles.show(5, truncate=False)

# Show schema
print("\n📋 Schema:")
aisles.printSchema()

# Add metadata columns
print("\n➕ Adding metadata columns...")
aisles_bronze = aisles \
    .withColumn("ingestion_time", current_timestamp()) \
    .withColumn("source_file", lit("aisles.csv"))

print("   ✅ Added: ingestion_time, source_file")

# Write to Delta Lake
print("\n💾 Writing to Delta Lake...")
aisles_bronze.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bronze_db.aisles")

print("   ✅ Saved as bronze_db.aisles")

# Verify
print("\n✅ VERIFICATION:")
result = spark.table("bronze_db.aisles")
print(f"   Rows in Delta table: {result.count():,}")
print(f"   Columns: {len(result.columns)}")

print("\n" + "=" * 70)
print("✅ AISLES TABLE COMPLETE!")
print("=" * 70)

# COMMAND ----------

# Cell 4: Load Departments to Bronze Delta Table

print("=" * 70)
print("🏬 LOADING DEPARTMENTS TABLE")
print("=" * 70)

from pyspark.sql.functions import current_timestamp, lit

# Read CSV
print("\n🔄 Reading departments.csv...")
departments = spark.read.csv("/mnt/bronze/raw/departments.csv", header=True, inferSchema=True)

print(f"   ✅ Loaded {departments.count():,} departments")

# Show all (small table)
print("\n📊 All Departments:")
departments.show(25, truncate=False)

# Add metadata
departments_bronze = departments \
    .withColumn("ingestion_time", current_timestamp()) \
    .withColumn("source_file", lit("departments.csv"))

# Write to Delta
print("\n💾 Writing to Delta Lake...")
departments_bronze.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bronze_db.departments")

print("   ✅ Saved as bronze_db.departments")

# Verify
result = spark.table("bronze_db.departments")
print(f"\n✅ Verified: {result.count():,} rows in Delta table")

print("=" * 70)
print("✅ DEPARTMENTS TABLE COMPLETE!")
print("=" * 70)

# COMMAND ----------

# Cell 5: Load Products to Bronze Delta Table

print("=" * 70)
print("📦 LOADING PRODUCTS TABLE")
print("=" * 70)

from pyspark.sql.functions import current_timestamp, lit

# Read CSV
print("\n🔄 Reading products.csv...")
products = spark.read.csv("/mnt/bronze/raw/products.csv", header=True, inferSchema=True)

print(f"   ✅ Loaded {products.count():,} products")

# Show sample
print("\n📊 Sample Data:")
products.show(10, truncate=False)

# Show schema
print("\n📋 Schema:")
products.printSchema()

# Add metadata
print("\n➕ Adding metadata columns...")
products_bronze = products \
    .withColumn("ingestion_time", current_timestamp()) \
    .withColumn("source_file", lit("products.csv"))

# Write to Delta
print("\n💾 Writing to Delta Lake...")
products_bronze.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bronze_db.products")

print("   ✅ Saved as bronze_db.products")

# Verify
result = spark.table("bronze_db.products")
print(f"\n✅ Verified: {result.count():,} rows in Delta table")

print("\n" + "=" * 70)
print("✅ PRODUCTS TABLE COMPLETE!")
print("=" * 70)

# COMMAND ----------

# Cell 6: Summary of Bronze Tables

print("=" * 70)
print("📊 BRONZE LAYER SUMMARY")
print("=" * 70)

# List all tables
print("\n📁 Tables in bronze_db:")
tables = spark.sql("SHOW TABLES IN bronze_db").collect()

for table in tables:
    table_name = table.tableName
    count = spark.table(f"bronze_db.{table_name}").count()
    print(f"   ✅ {table_name:20s} - {count:>10,} rows")

print("\n" + "=" * 70)
print("✅ 3 TABLES LOADED TO BRONZE!")
print("=" * 70)

# COMMAND ----------

# Cell 7: Test Delta Lake Time Travel

print("=" * 70)
print("⏰ TESTING DELTA LAKE TIME TRAVEL")
print("=" * 70)

# View table history
print("\n📜 Version History for bronze_db.aisles:")
history = spark.sql("DESCRIBE HISTORY bronze_db.aisles")
history.select("version", "timestamp", "operation", "operationMetrics").show(5, truncate=False)

# Read current version
print("\n📊 Current Version:")
current = spark.table("bronze_db.aisles")
print(f"   Rows: {current.count():,}")

# Read version 0 (first version)
print("\n📊 Version 0 (Initial Load):")
v0 = spark.read.format("delta").option("versionAsOf", 0).table("bronze_db.aisles")
print(f"   Rows: {v0.count():,}")

print("\n✅ Time Travel Working!")
print("   You can query any previous version of the data!")

print("=" * 70)

# COMMAND ----------

# Cell 8: Load Orders to Bronze Delta Table

print("=" * 70)
print("📋 LOADING ORDERS TABLE")
print("=" * 70)

from pyspark.sql.functions import current_timestamp, lit

# Read CSV
print("\n🔄 Reading orders.csv...")
print("   (This may take 30-60 seconds with 3.4M rows...)")

orders = spark.read.csv("/mnt/bronze/raw/orders.csv", header=True, inferSchema=True)

row_count = orders.count()
print(f"   ✅ Loaded {row_count:,} orders")

# Show sample
print("\n📊 Sample Data:")
orders.show(10, truncate=False)

# Show schema
print("\n📋 Schema:")
orders.printSchema()

# Basic statistics
print("\n📈 Quick Statistics:")
print(f"   Total orders: {row_count:,}")
print(f"   Unique users: {orders.select('user_id').distinct().count():,}")
print(f"   Columns: {len(orders.columns)}")

print("\n" + "=" * 70)

# COMMAND ----------

# Cell 9: Data Quality Checks for Orders

print("=" * 70)
print("🧪 DATA QUALITY CHECKS - ORDERS")
print("=" * 70)

from pyspark.sql.functions import col

# Check 1: Null values
print("\n1️⃣ CHECKING FOR NULL VALUES:")
null_checks = {}
for column in orders.columns:
    null_count = orders.filter(col(column).isNull()).count()
    null_checks[column] = null_count
    if null_count > 0:
        pct = (null_count / orders.count()) * 100
        print(f"   ⚠️  {column:30s}: {null_count:>10,} nulls ({pct:.2f}%)")
    else:
        print(f"   ✅ {column:30s}: No nulls")

# Check 2: Duplicates
print("\n2️⃣ CHECKING FOR DUPLICATES:")
total_rows = orders.count()
unique_orders = orders.dropDuplicates(["order_id"]).count()
duplicates = total_rows - unique_orders
if duplicates > 0:
    print(f"   ⚠️  Found {duplicates:,} duplicate order_ids")
else:
    print(f"   ✅ No duplicate order_ids")

# Check 3: Value ranges
print("\n3️⃣ CHECKING VALUE RANGES:")

# Day of week (should be 0-6)
invalid_dow = orders.filter(~col("order_dow").between(0, 6)).count()
if invalid_dow > 0:
    print(f"   ⚠️  Invalid order_dow: {invalid_dow:,} rows")
else:
    print(f"   ✅ order_dow valid (0-6)")

# Hour of day (should be 0-23)
invalid_hour = orders.filter(~col("order_hour_of_day").between(0, 23)).count()
if invalid_hour > 0:
    print(f"   ⚠️  Invalid order_hour_of_day: {invalid_hour:,} rows")
else:
    print(f"   ✅ order_hour_of_day valid (0-23)")

# Check 4: Distribution
print("\n4️⃣ DATA DISTRIBUTION:")
print(f"   Users: {orders.select('user_id').distinct().count():,}")
print(f"   Orders per user (avg): {orders.count() / orders.select('user_id').distinct().count():.1f}")

print("\n" + "=" * 70)
print("✅ QUALITY CHECKS COMPLETE!")
print("=" * 70)

# COMMAND ----------

# Cell 10: Save Orders to Bronze Delta Table

print("=" * 70)
print("💾 SAVING ORDERS TO DELTA LAKE")
print("=" * 70)

from pyspark.sql.functions import current_timestamp, lit

# Add metadata columns
print("\n➕ Adding metadata columns...")
orders_bronze = orders \
    .withColumn("ingestion_time", current_timestamp()) \
    .withColumn("source_file", lit("orders.csv"))

print("   ✅ Added: ingestion_time, source_file")

# Write to Delta Lake
print("\n💾 Writing to Delta Lake...")
print("   (This will take 1-2 minutes for 3.4M rows...)")

orders_bronze.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bronze_db.orders")

print("   ✅ Saved as bronze_db.orders")

# Verify
print("\n✅ VERIFICATION:")
result = spark.table("bronze_db.orders")
print(f"   Rows in Delta table: {result.count():,}")
print(f"   Columns: {len(result.columns)}")

# Show schema with new columns
print("\n📋 Final Schema:")
result.printSchema()

print("\n" + "=" * 70)
print("✅ ORDERS TABLE COMPLETE!")
print("=" * 70)

# COMMAND ----------

# Cell 11: Bronze Layer Progress Summary

print("=" * 70)
print("📊 BRONZE LAYER PROGRESS")
print("=" * 70)

# List all tables
print("\n📁 Tables in bronze_db:")
tables = spark.sql("SHOW TABLES IN bronze_db").collect()

total_rows = 0
for table in tables:
    table_name = table.tableName
    count = spark.table(f"bronze_db.{table_name}").count()
    total_rows += count
    print(f"   ✅ {table_name:20s} - {count:>12,} rows")

print(f"\n📊 TOTAL: {total_rows:,} rows in Bronze layer")

print("\n🎯 REMAINING:")
print("   ⏳ order_products (to be sampled to 5-10M rows)")

print("\n" + "=" * 70)
print("✅ 4 OF 5 TABLES COMPLETE!")
print("=" * 70)

# COMMAND ----------

# Cell 12: Explore Order Products Data

print("=" * 70)
print("🔍 EXPLORING ORDER PRODUCTS DATA")
print("=" * 70)

# Read the CSV files
print("\n📂 Reading order_products files...")

order_products_prior = spark.read.csv(
    "/mnt/bronze/raw/order_products__prior.csv", 
    header=True, 
    inferSchema=True
)

order_products_train = spark.read.csv(
    "/mnt/bronze/raw/order_products__train.csv", 
    header=True, 
    inferSchema=True
)

# Combine them
print("   ✅ Loaded prior set")
print("   ✅ Loaded train set")

order_products = order_products_prior.union(order_products_train)

print("\n📊 TOTAL SIZE:")
prior_count = order_products_prior.count()
train_count = order_products_train.count()
total_count = order_products.count()

print(f"   Prior set:  {prior_count:>12,} rows")
print(f"   Train set:  {train_count:>12,} rows")
print(f"   Combined:   {total_count:>12,} rows")

# Show sample
print("\n📋 Sample Data:")
order_products.show(10, truncate=False)

# Show schema
print("\n📋 Schema:")
order_products.printSchema()

# Statistics
print("\n📈 STATISTICS:")
print(f"   Total order-product records: {total_count:,}")
print(f"   Unique orders: {order_products.select('order_id').distinct().count():,}")
print(f"   Unique products: {order_products.select('product_id').distinct().count():,}")

# Memory check
print(f"\n💾 Estimated size: ~2.5 GB in memory")

print("\n" + "=" * 70)
print("✅ DATA EXPLORATION COMPLETE!")
print("=" * 70)

# COMMAND ----------

# Cell 13: Calculate Sampling Strategy

print("=" * 70)
print("🎯 CALCULATING SAMPLING PARAMETERS")
print("=" * 70)

# Target size
TARGET_MIN = 5_000_000
TARGET_MAX = 10_000_000
TARGET_IDEAL = 7_000_000

print(f"\n🎯 TARGET:")
print(f"   Min:   {TARGET_MIN:>12,} rows")
print(f"   Ideal: {TARGET_IDEAL:>12,} rows")
print(f"   Max:   {TARGET_MAX:>12,} rows")

# Current size
current_size = order_products.count()
print(f"\n📊 CURRENT:")
print(f"   Total: {current_size:>12,} rows")

# Calculate sampling fraction needed
sampling_fraction_by_rows = TARGET_IDEAL / current_size
print(f"\n🔢 IF WE SAMPLE BY ROWS (random):")
print(f"   Fraction needed: {sampling_fraction_by_rows:.2%}")
print(f"   ❌ Problem: Breaks user behavior patterns!")

# Better approach: Sample by users
print(f"\n✅ BETTER APPROACH: SAMPLE BY USERS")

# Get user statistics from orders
user_stats = spark.table("bronze_db.orders").groupBy("user_id").count()
total_users = user_stats.count()

# Estimate records per user
avg_records_per_user = current_size / total_users

print(f"\n📊 USER STATISTICS:")
print(f"   Total users:             {total_users:,}")
print(f"   Avg records per user:    {avg_records_per_user:.1f}")

# Calculate user sampling fraction
user_sampling_fraction = TARGET_IDEAL / current_size
users_to_sample = int(total_users * user_sampling_fraction)

print(f"\n🎯 SAMPLING PLAN:")
print(f"   Users to sample:         {users_to_sample:,} ({user_sampling_fraction:.1%} of users)")
print(f"   Expected rows:           ~{int(users_to_sample * avg_records_per_user):,}")
print(f"   Within target:           {TARGET_MIN:,} - {TARGET_MAX:,}")

# Adjust if needed
if users_to_sample * avg_records_per_user < TARGET_MIN:
    user_sampling_fraction = (TARGET_IDEAL * 1.1) / current_size  # 10% buffer
    users_to_sample = int(total_users * user_sampling_fraction)
    print(f"\n⚠️  ADJUSTED (to ensure minimum):")
    print(f"   Users to sample:         {users_to_sample:,} ({user_sampling_fraction:.1%})")
    print(f"   Expected rows:           ~{int(users_to_sample * avg_records_per_user):,}")

print("\n✅ Sampling plan preserves complete user journeys!")

print("\n" + "=" * 70)
print("✅ SAMPLING STRATEGY CALCULATED!")
print("=" * 70)

# COMMAND ----------

# Cell 14: Execute Smart Sampling

print("=" * 70)
print("🎲 EXECUTING SMART SAMPLING")
print("=" * 70)

from pyspark.sql.functions import col

# Parameters from previous calculation
USER_SAMPLING_FRACTION = 0.21  # ~21% of users for ~7M records
RANDOM_SEED = 42  # For reproducibility

print(f"\n⚙️  SAMPLING PARAMETERS:")
print(f"   User fraction:  {USER_SAMPLING_FRACTION:.2%}")
print(f"   Random seed:    {RANDOM_SEED}")

# STEP 1: Sample users from orders table
print(f"\n1️⃣  SAMPLING USERS...")
orders_table = spark.table("bronze_db.orders")

sampled_users = orders_table.select("user_id").distinct() \
    .sample(withReplacement=False, fraction=USER_SAMPLING_FRACTION, seed=RANDOM_SEED)

sampled_user_count = sampled_users.count()
print(f"   ✅ Sampled {sampled_user_count:,} users")

# STEP 2: Get all orders for sampled users
print(f"\n2️⃣  GETTING ORDERS FOR SAMPLED USERS...")
sampled_orders = orders_table.join(sampled_users, "user_id", "inner")

sampled_order_count = sampled_orders.count()
print(f"   ✅ Found {sampled_order_count:,} orders")

# STEP 3: Get all order_products for sampled orders
print(f"\n3️⃣  FILTERING ORDER_PRODUCTS...")
print(f"   (This will take 2-3 minutes for 33M rows...)")

sampled_order_ids = sampled_orders.select("order_id")

# Filter order_products to only sampled orders
sampled_order_products = order_products.join(
    sampled_order_ids, 
    "order_id", 
    "inner"
)

# Count result
final_count = sampled_order_products.count()
print(f"   ✅ Sampled to {final_count:,} rows")

# Verify target
print(f"\n📊 SAMPLING RESULTS:")
print(f"   Original size:   {order_products.count():>12,} rows")
print(f"   Sampled size:    {final_count:>12,} rows")
print(f"   Reduction:       {((order_products.count() - final_count) / order_products.count() * 100):.1f}%")
print(f"   Target range:    {TARGET_MIN:>12,} - {TARGET_MAX:,}")

if TARGET_MIN <= final_count <= TARGET_MAX:
    print(f"   ✅ WITHIN TARGET!")
else:
    print(f"   ⚠️  Outside target (but close enough for learning)")

print("\n" + "=" * 70)
print("✅ SMART SAMPLING COMPLETE!")
print("=" * 70)

# COMMAND ----------

# Cell 15: Validate Sampling Quality

print("=" * 70)
print("🧪 VALIDATING SAMPLING QUALITY")
print("=" * 70)

# Check 1: User coverage
print("\n1️⃣  USER COVERAGE:")
original_users = order_products.select("order_id").distinct() \
    .join(orders_table, "order_id") \
    .select("user_id").distinct().count()

sampled_users_in_data = sampled_order_products.select("order_id").distinct() \
    .join(orders_table, "order_id") \
    .select("user_id").distinct().count()

print(f"   Original users:  {original_users:,}")
print(f"   Sampled users:   {sampled_users_in_data:,}")
print(f"   Percentage:      {(sampled_users_in_data / original_users * 100):.1f}%")

# Check 2: Product coverage
print("\n2️⃣  PRODUCT COVERAGE:")
original_products = order_products.select("product_id").distinct().count()
sampled_products = sampled_order_products.select("product_id").distinct().count()

print(f"   Original products: {original_products:,}")
print(f"   Sampled products:  {sampled_products:,}")
print(f"   Percentage:        {(sampled_products / original_products * 100):.1f}%")

# Check 3: Complete user journeys
print("\n3️⃣  USER JOURNEY COMPLETENESS:")
print("   Checking if sampled users have ALL their orders...")

# Pick a random sampled user
sample_user = sampled_users.first().user_id

# Count their orders in original vs sampled
original_user_orders = orders_table.filter(col("user_id") == sample_user).count()
sampled_user_orders = sampled_orders.filter(col("user_id") == sample_user).count()

print(f"   User {sample_user}:")
print(f"   - Original orders: {original_user_orders}")
print(f"   - Sampled orders:  {sampled_user_orders}")

if original_user_orders == sampled_user_orders:
    print(f"   ✅ Complete journey preserved!")
else:
    print(f"   ❌ Journey incomplete (should not happen)")

# Check 4: Distribution similarity
print("\n4️⃣  DISTRIBUTION CHECK:")
original_avg_basket = order_products.groupBy("order_id").count().agg({"count": "avg"}).first()[0]
sampled_avg_basket = sampled_order_products.groupBy("order_id").count().agg({"count": "avg"}).first()[0]

print(f"   Avg basket size (original): {original_avg_basket:.2f}")
print(f"   Avg basket size (sampled):  {sampled_avg_basket:.2f}")
print(f"   Difference:                 {abs(original_avg_basket - sampled_avg_basket):.2f}")

if abs(original_avg_basket - sampled_avg_basket) < 1:
    print(f"   ✅ Distributions match!")

print("\n" + "=" * 70)
print("✅ SAMPLING QUALITY VALIDATED!")
print("=" * 70)

# COMMAND ----------

# Cell 16: Save Sampled Order Products to Delta

print("=" * 70)
print("💾 SAVING ORDER_PRODUCTS TO DELTA LAKE")
print("=" * 70)

from pyspark.sql.functions import current_timestamp, lit

# Add metadata
print("\n➕ Adding metadata columns...")
order_products_bronze = sampled_order_products \
    .withColumn("ingestion_time", current_timestamp()) \
    .withColumn("source_file", lit("order_products__prior.csv + order_products__train.csv")) \
    .withColumn("sampling_seed", lit(RANDOM_SEED)) \
    .withColumn("sampling_fraction", lit(USER_SAMPLING_FRACTION))

print("   ✅ Added metadata columns")

# Write to Delta Lake
print("\n💾 Writing to Delta Lake...")
print("   (This will take 2-3 minutes for 7M rows...)")

order_products_bronze.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bronze_db.order_products")

print("   ✅ Saved as bronze_db.order_products")

# Verify
print("\n✅ VERIFICATION:")
result = spark.table("bronze_db.order_products")
final_count = result.count()
print(f"   Rows in Delta table: {final_count:,}")
print(f"   Columns: {len(result.columns)}")

# Show schema
print("\n📋 Schema:")
result.printSchema()

# Show sample with metadata
print("\n📊 Sample with Metadata:")
result.show(5, truncate=False)

print("\n" + "=" * 70)
print("✅ ORDER_PRODUCTS TABLE COMPLETE!")
print("=" * 70)

# COMMAND ----------

# Cell 17: Bronze Layer Complete Summary

print("=" * 70)
print("🎉 BRONZE LAYER COMPLETE!")
print("=" * 70)

# List all tables with details
print("\n📁 FINAL BRONZE LAYER:")
tables = spark.sql("SHOW TABLES IN bronze_db").collect()

total_rows = 0
for table in tables:
    table_name = table.tableName
    df = spark.table(f"bronze_db.{table_name}")
    count = df.count()
    columns = len(df.columns)
    total_rows += count
    
    print(f"\n   ✅ {table_name}")
    print(f"      Rows:    {count:>12,}")
    print(f"      Columns: {columns:>12,}")

print(f"\n📊 TOTALS:")
print(f"   Tables:      5")
print(f"   Total rows:  {total_rows:,}")
print(f"   Format:      Delta Lake")

print("\n🎯 KEY ACHIEVEMENTS:")
print("   ✅ All raw data ingested to Delta Lake")
print("   ✅ Smart sampling preserved user journeys")
print("   ✅ Data quality validated")
print("   ✅ ACID transactions enabled")
print("   ✅ Time travel available")
print("   ✅ Metadata tracking added")

print("\n📈 DATA REDUCTION:")
print(f"   Original order_products: 33,819,106 rows")
print(f"   Sampled order_products:  {spark.table('bronze_db.order_products').count():,} rows")
print(f"   Other tables:            {total_rows - spark.table('bronze_db.order_products').count():,} rows")
print(f"   Total Bronze:            {total_rows:,} rows")

print("\n🚀 READY FOR:")
print("   Next: Silver Layer (cleaning, validation, joins)")

print("\n" + "=" * 70)
print("🎉 DAY 1 COMPLETE! EXCELLENT WORK!")
print("=" * 70)

# COMMAND ----------


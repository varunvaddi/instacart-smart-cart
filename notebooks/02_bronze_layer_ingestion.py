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
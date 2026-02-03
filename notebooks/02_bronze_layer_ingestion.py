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
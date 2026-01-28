# Databricks notebook source
# MAGIC %md
# MAGIC #Cell 1: Test Cluster & Libraries

# COMMAND ----------

# Cell 1: Verify cluster and libraries

print("=" * 70)
print("🚀 CLUSTER & ENVIRONMENT CHECK")
print("=" * 70)

# Cluster info
print(f"\n📊 CLUSTER:")
print(f"   Spark Version: {spark.version}")
print(f"   Python Version: 3.10+")

# Test core libraries
print(f"\n📚 LIBRARIES:")

import mlflow
print(f"   ✓ MLflow: {mlflow.__version__}")

import pandas as pd
import numpy as np
print(f"   ✓ Pandas: {pd.__version__}")
print(f"   ✓ NumPy: {np.__version__}")

from pyspark.sql import functions as F
print(f"   ✓ PySpark SQL: Available")

# Quick Spark test
test_df = spark.range(1000000)
count = test_df.count()
print(f"\n🧪 SPARK TEST:")
print(f"   Created and counted {count:,} rows")
print(f"   ✓ Spark is working!")

print("\n" + "=" * 70)
print("✅ CLUSTER READY!")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC #Cell 2: Test Key Vault Connection

# COMMAND ----------

dbutils.secrets.list("key-vault")
# Should return ['adls-account-key']

storage_key = dbutils.secrets.get("key-vault", "adls-account-key")


# COMMAND ----------


print("=" * 70)
print("🔐 TESTING KEY VAULT CONNECTION")
print("=" * 70)

try:
    # Retrieve secret from Key Vault via secret scope
    storage_key = dbutils.secrets.get(scope="key-vault", key="adls-account-key")
    
    print("\n✅ SUCCESS!")
    print(f"   Secret scope: key-vault")
    print(f"   Secret key: adls-account-key")
    print(f"   Retrieved successfully!")
    print(f"   Key length: {len(storage_key)} characters")
    print(f"   First 4 chars: {storage_key[:4]}...")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure secret scope 'key-vault' exists")
    print("2. Make sure secret 'adls-account-key' exists in Key Vault")
    print("3. Check Databricks has permission to access Key Vault")
    raise

print("\n" + "=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC #Cell 3: Mount ADLS Storage

# COMMAND ----------

# Cell 3: Mount ADLS (using WASBS protocol)

print("=" * 70)
print("🔗 MOUNTING AZURE DATA LAKE STORAGE GEN2")
print("=" * 70)

storage_account_name = "instacartdatalake"
container_name = "bronze-data"
mount_point = "/mnt/bronze"

print(f"\n📦 Configuration:")
print(f"   Storage Account: {storage_account_name}")
print(f"   Container: {container_name}")
print(f"   Mount Point: {mount_point}")

# Check if already mounted
print(f"\n🔍 Checking existing mounts...")
existing_mounts = [mount.mountPoint for mount in dbutils.fs.mounts()]

if mount_point in existing_mounts:
    print(f"   ⚠️  {mount_point} already mounted, unmounting...")
    dbutils.fs.unmount(mount_point)
    print(f"   ✓ Unmounted")
else:
    print(f"   ✓ Not currently mounted")

# Get storage key from Key Vault
print(f"\n🔐 Retrieving storage key...")
storage_account_key = dbutils.secrets.get(scope="key-vault", key="adls-account-key")
print(f"   ✓ Retrieved storage key")

# Configure mount with WASBS protocol (more compatible)
configs = {
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_key
}

# Mount using WASBS protocol
print(f"\n🔧 Mounting storage...")
dbutils.fs.mount(
    source=f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
    mount_point=mount_point,
    extra_configs=configs
)

print(f"   ✓ Mount successful!")

# Verify mount
print(f"\n✅ Verification:")
files = dbutils.fs.ls(mount_point)
print(f"   Contents of {mount_point}:")
for file in files:
    print(f"      📁 {file.name}")

print("\n" + "=" * 70)
print("✅ ADLS MOUNTED SUCCESSFULLY!")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC #Cell 4: List Data Files

# COMMAND ----------

# Cell 4: Explore raw data files

print("=" * 70)
print("📂 RAW DATA FILES")
print("=" * 70)

# List files
files = dbutils.fs.ls("/mnt/bronze/raw/")

print(f"\n📊 Files in /mnt/bronze/raw/:\n")

total_size = 0
for file in files:
    size_mb = file.size / (1024 * 1024)
    total_size += size_mb
    print(f"   {file.name:40s} {size_mb:10.2f} MB")

print(f"\n{'─' * 70}")
print(f"   {'TOTAL':40s} {total_size:10.2f} MB")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC #Cell 5: Read Your First Data with Spark!

# COMMAND ----------

# Cell 5: Read orders data with PySpark

print("=" * 70)
print("📖 READING ORDERS WITH PYSPARK")
print("=" * 70)

# Read CSV with Spark
print("\n🔄 Loading orders.csv...")
orders = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/mnt/bronze/raw/orders.csv")

print("   ✓ Loaded into Spark DataFrame")

# Show schema
print("\n📋 Schema:")
orders.printSchema()

# Show sample data
print("\n📊 Sample Data:")
orders.show(10, truncate=False)

# Statistics
print("\n📈 Statistics:")
row_count = orders.count()
user_count = orders.select("user_id").distinct().count()

print(f"   Total orders: {row_count:,}")
print(f"   Unique users: {user_count:,}")
print(f"   Columns: {len(orders.columns)}")

print("\n" + "=" * 70)
print("🎉 YOU JUST PROCESSED 3.4M ROWS WITH SPARK!")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC
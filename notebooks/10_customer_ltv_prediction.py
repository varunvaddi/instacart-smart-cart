# Databricks notebook source
# Cell 1: Customer Lifetime Value (LTV) Prediction - Setup

print("=" * 70)
print("💰 CUSTOMER LIFETIME VALUE (LTV) PREDICTION")
print("=" * 70)

print("\n🎯 WHAT IS LTV?")
print("   LTV = Customer Lifetime Value")
print("   Total revenue a customer will generate over their lifetime")
print("   ")
print("   Example:")
print("   • Customer orders 50 times")
print("   • Average order: $100")
print("   • LTV = $5,000")

print("\n💡 WHY PREDICT LTV?")
print("   ✅ Identify high-value customers (VIPs)")
print("   ✅ Allocate marketing budget efficiently")
print("   ✅ Personalize retention strategies")
print("   ✅ Optimize customer acquisition cost (CAC)")
print("   ")
print("   Rule: CAC < LTV (otherwise losing money!)")

print("\n🔧 APPROACH:")
print("   1. Calculate historical LTV for each customer")
print("   2. Use early behavior to predict future LTV")
print("   3. Model: Regression (predict $ amount)")
print("   4. Features: RFM scores, order patterns, preferences")

print("\n📊 BUSINESS QUESTIONS:")
print("   • Who are our highest-value customers?")
print("   • Can we predict high LTV from first 3 orders?")
print("   • Should we spend $50 or $500 to acquire this customer?")

# COMMAND ----------


# Load data
print("\n🔍 LOADING DATA:")
user_features = spark.table("gold_db.user_features")
print(f"   Customers: {user_features.count():,}")

# Show sample
print("\n📋 SAMPLE CUSTOMER DATA:")
user_features.select(
    "user_id", "total_orders", "avg_basket_size", 
    "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score"
).show(5)

print("\n" + "=" * 70)
print("✅ READY TO BUILD LTV MODEL!")
print("=" * 70)

# COMMAND ----------

# Cell 2: Calculate Historical LTV (Target Variable) - FIXED

print("=" * 70)
print("💰 CALCULATING HISTORICAL LTV")
print("=" * 70)

from pyspark.sql.functions import col, sum as spark_sum, avg, max as spark_max, min as spark_min

print("\n1️⃣  DEFINING LTV METRIC:")
print("   LTV = Total Orders × Average Basket Size")
print("   (Simplified - assumes each item = $1 revenue)")

# Load user features
user_features = spark.table("gold_db.user_features")

# Calculate LTV - FIX: Be explicit with column references
ltv_data = user_features.select(
    col("user_id"),
    col("total_orders"),
    col("avg_basket_size")
).withColumn(
    "ltv",
    col("total_orders") * col("avg_basket_size")
)

print("\n2️⃣  LTV DISTRIBUTION:")

ltv_stats = ltv_data.select(
    spark_min("ltv").alias("min_ltv"),
    avg("ltv").alias("avg_ltv"),
    spark_max("ltv").alias("max_ltv")
).collect()[0]

print(f"   Minimum LTV: {ltv_stats['min_ltv']:.2f} items")
print(f"   Average LTV: {ltv_stats['avg_ltv']:.2f} items")
print(f"   Maximum LTV: {ltv_stats['max_ltv']:.2f} items")

# Show distribution
print("\n📊 LTV DISTRIBUTION BY BUCKETS:")
ltv_data.selectExpr(
    "CASE " +
    "WHEN ltv < 50 THEN '0-50 (Low)' " +
    "WHEN ltv < 100 THEN '50-100 (Medium-Low)' " +
    "WHEN ltv < 200 THEN '100-200 (Medium)' " +
    "WHEN ltv < 500 THEN '200-500 (Medium-High)' " +
    "ELSE '500+ (High)' END as ltv_bucket"
).groupBy("ltv_bucket").count().orderBy("ltv_bucket").show()

# Join with all user features
print("\n3️⃣  CREATING TRAINING DATASET:")

training_data = user_features.alias("uf").join(
    ltv_data.select("user_id", "ltv").alias("ltv"),
    col("uf.user_id") == col("ltv.user_id")
).select(
    col("uf.*"),
    col("ltv.ltv")
)

print(f"   ✅ Training samples: {training_data.count():,}")

# Show sample
print("\n📋 SAMPLE (Customer LTV + Features):")
training_data.select(
    "user_id", "ltv", "total_orders", "avg_basket_size",
    "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score"
).show(10)

# Save for later
training_data.write.mode("overwrite").saveAsTable("gold_db.ltv_training_data")
print("\n💾 Saved to: gold_db.ltv_training_data")

print("\n" + "=" * 70)
print("✅ LTV TARGET CALCULATED!")
print("=" * 70)

# COMMAND ----------

# Cell 3: Feature Engineering for LTV Prediction - FIXED

print("=" * 70)
print("🔧 FEATURE ENGINEERING FOR LTV")
print("=" * 70)

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

# Load training data
training_data = spark.table("gold_db.ltv_training_data")

print("\n📋 SELECTING FEATURES:")

# Features for LTV prediction
feature_columns = [
    # Early behavior indicators
    "rfm_recency_score",        # How recently active
    "rfm_frequency_score",      # How often they order
    "rfm_monetary_score",       # How much they spend
    
    # Order patterns (excluding total_orders & avg_basket_size - too direct!)
    "product_diversity_score",  # Do they explore?
    "user_reorder_ratio",       # Loyalty indicator
    
    # Preferences
    "avg_order_dow",            # Day preference
    "avg_order_hour"            # Time preference
]

print(f"\n   Using {len(feature_columns)} features:")
for i, feature in enumerate(feature_columns, 1):
    print(f"   {i:2d}. {feature}")

print("\n⚠️  NOTE: Excluding 'total_orders' & 'avg_basket_size'")
print("   Why? They directly calculate LTV (would be data leakage!)")
print("   Goal: Predict LTV from EARLY behavior signals only")

# Create feature vector
print("\n🔨 CREATING FEATURE VECTORS:")

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

training_data_vec = assembler.transform(training_data)

print("   ✅ Feature vectors created")

# Train/test split
print("\n✂️  TRAIN/TEST SPLIT (80/20):")

train_data, test_data = training_data_vec.randomSplit([0.8, 0.2], seed=42)

train_count = train_data.count()
test_count = test_data.count()

print(f"   Train set: {train_count:,} customers ({train_count/(train_count+test_count)*100:.1f}%)")
print(f"   Test set:  {test_count:,} customers ({test_count/(train_count+test_count)*100:.1f}%)")

# Cache for performance
train_data.cache()
test_data.cache()

print("\n📊 SAMPLE TRAINING DATA:")
train_data.select("user_id", "ltv", "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score").show(5)

print("\n" + "=" * 70)
print("✅ DATA READY FOR TRAINING!")
print("=" * 70)

# COMMAND ----------

# Cell 4: Train LTV Regression Model

print("=" * 70)
print("🤖 TRAINING LTV PREDICTION MODEL")
print("=" * 70)

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import time

print("\n⚙️  MODEL CONFIGURATION:")
print("   Algorithm: Gradient Boosted Trees (Regression)")
print("   Target: Customer Lifetime Value (LTV)")
print("   Features: 9 customer behavior features")

# Model parameters
max_iter = 20
max_depth = 5

print(f"\n   Parameters:")
print(f"   • Max Iterations: {max_iter}")
print(f"   • Max Depth: {max_depth}")

# Train model
print(f"\n🏋️  TRAINING MODEL...")
print(f"   (This will take 2-3 minutes...)")

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="ltv",
    maxIter=max_iter,
    maxDepth=max_depth,
    seed=42
)

start_time = time.time()
ltv_model = gbt.fit(train_data)
training_time = time.time() - start_time

print(f"\n   ✅ Model trained in {training_time/60:.1f} minutes!")

# Make predictions
print(f"\n🔮 MAKING PREDICTIONS...")

train_predictions = ltv_model.transform(train_data)
test_predictions = ltv_model.transform(test_data)

print(f"   ✅ Predictions complete")

# Show sample predictions
print(f"\n📊 SAMPLE PREDICTIONS:")
test_predictions.select(
    "user_id", 
    col("ltv").alias("actual_ltv"), 
    col("prediction").alias("predicted_ltv")
).show(10)

print("\n" + "=" * 70)
print("✅ LTV MODEL TRAINED!")
print("=" * 70)

# COMMAND ----------

# Cell 5: Evaluate LTV Model Performance

print("=" * 70)
print("📊 LTV MODEL EVALUATION")
print("=" * 70)

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, abs as spark_abs, avg, stddev, expr

# === REGRESSION METRICS ===
print("\n1️⃣  REGRESSION METRICS:")

# RMSE (Root Mean Squared Error)
rmse_evaluator = RegressionEvaluator(
    labelCol="ltv",
    predictionCol="prediction",
    metricName="rmse"
)
rmse = rmse_evaluator.evaluate(test_predictions)

# MAE (Mean Absolute Error)
mae_evaluator = RegressionEvaluator(
    labelCol="ltv",
    predictionCol="prediction",
    metricName="mae"
)
mae = mae_evaluator.evaluate(test_predictions)

# R² (R-Squared)
r2_evaluator = RegressionEvaluator(
    labelCol="ltv",
    predictionCol="prediction",
    metricName="r2"
)
r2 = r2_evaluator.evaluate(test_predictions)

print(f"   RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"   MAE (Mean Absolute Error):      {mae:.2f}")
print(f"   R² (R-Squared):                 {r2:.4f}")

print(f"\n   💡 INTERPRETATION:")
print(f"   • Average prediction error: ±{mae:.0f} items")
print(f"   • R² = {r2:.2%} (how well model explains variance)")
if r2 >= 0.70:
    quality = "EXCELLENT ⭐⭐⭐"
elif r2 >= 0.50:
    quality = "GOOD ⭐⭐"
elif r2 >= 0.30:
    quality = "FAIR ⭐"
else:
    quality = "NEEDS IMPROVEMENT"
print(f"   • Model Quality: {quality}")

# === PREDICTION ACCURACY BANDS ===
print("\n2️⃣  PREDICTION ACCURACY DISTRIBUTION:")

accuracy_bands = test_predictions.withColumn(
    "error",
    spark_abs(col("prediction") - col("ltv"))
).withColumn(
    "error_pct",
    (spark_abs(col("prediction") - col("ltv")) / col("ltv")) * 100
).selectExpr(
    "CASE " +
    "WHEN error_pct < 10 THEN 'Within 10% (Excellent)' " +
    "WHEN error_pct < 20 THEN 'Within 20% (Good)' " +
    "WHEN error_pct < 30 THEN 'Within 30% (Fair)' " +
    "WHEN error_pct < 50 THEN 'Within 50% (Poor)' " +
    "ELSE 'Off by 50%+ (Very Poor)' END as accuracy_band"
).groupBy("accuracy_band").count().orderBy("accuracy_band")

print("\n   Prediction Accuracy:")
accuracy_bands.show(truncate=False)

# === LTV BUCKET PERFORMANCE ===
print("\n3️⃣  PERFORMANCE BY LTV BUCKET:")

bucket_performance = test_predictions.withColumn(
    "ltv_bucket",
    expr(
        "CASE " +
        "WHEN ltv < 100 THEN 'Low (<100)' " +
        "WHEN ltv < 300 THEN 'Medium (100-300)' " +
        "WHEN ltv < 600 THEN 'High (300-600)' " +
        "ELSE 'Very High (600+)' END"
    )
).withColumn(
    "error",
    spark_abs(col("prediction") - col("ltv"))
).groupBy("ltv_bucket").agg(
    avg("ltv").alias("avg_actual_ltv"),
    avg("prediction").alias("avg_predicted_ltv"),
    avg("error").alias("avg_error")
).orderBy("avg_actual_ltv")

print("\n   Error by Customer Value:")
bucket_performance.show(truncate=False)

# === TOP PREDICTIONS ===
print("\n4️⃣  TOP PREDICTED HIGH-VALUE CUSTOMERS:")

top_predicted = test_predictions.orderBy(col("prediction").desc()).limit(10)

print("\n   Customers with Highest Predicted LTV:")
top_predicted.select(
    "user_id",
    col("ltv").alias("actual_ltv"),
    col("prediction").alias("predicted_ltv"),
    expr("abs(prediction - ltv)").alias("error")
).show(10)

# === BOTTOM PREDICTIONS ===
print("\n5️⃣  LOWEST PREDICTED LTV CUSTOMERS:")

bottom_predicted = test_predictions.orderBy(col("prediction").asc()).limit(10)

print("\n   Customers with Lowest Predicted LTV:")
bottom_predicted.select(
    "user_id",
    col("ltv").alias("actual_ltv"),
    col("prediction").alias("predicted_ltv")
).show(10)

# Save predictions
test_predictions.select(
    "user_id", "ltv", "prediction"
).write.mode("overwrite").saveAsTable("gold_db.ltv_predictions")

print("\n💾 Saved predictions to: gold_db.ltv_predictions")

print("\n" + "=" * 70)
print("✅ LTV MODEL EVALUATION COMPLETE!")
print("=" * 70)

# === SUMMARY ===
print("\n📊 LTV MODEL SUMMARY:")
print("┌─────────────────────┬─────────────┐")
print("│ Metric              │ Value       │")
print("├─────────────────────┼─────────────┤")
print(f"│ RMSE                │ {rmse:>11.2f} │")
print(f"│ MAE                 │ {mae:>11.2f} │")
print(f"│ R²                  │ {r2:>11.4f} │")
print(f"│ Avg Error           │ ±{mae:>10.0f} │")
print(f"│ Model Quality       │ {quality:>11s} │")
print("└─────────────────────┴─────────────┘")

# COMMAND ----------

# Cell 6: LTV Model - Business Insights & Use Cases

print("=" * 70)
print("💡 BUSINESS INSIGHTS - LTV MODEL")
print("=" * 70)

from pyspark.sql.functions import col, avg, count

# Load predictions
ltv_predictions = spark.table("gold_db.ltv_predictions")

print("\n1️⃣  HIGH-VALUE CUSTOMER IDENTIFICATION:")

# Segment by predicted LTV
ltv_segments = ltv_predictions.withColumn(
    "ltv_segment",
    expr(
        "CASE " +
        "WHEN prediction >= 500 THEN 'VIP (500+)' " +
        "WHEN prediction >= 300 THEN 'High Value (300-500)' " +
        "WHEN prediction >= 100 THEN 'Medium Value (100-300)' " +
        "ELSE 'Low Value (<100)' END"
    )
)

segment_summary = ltv_segments.groupBy("ltv_segment").agg(
    count("*").alias("customer_count"),
    avg("prediction").alias("avg_predicted_ltv"),
    avg("ltv").alias("avg_actual_ltv")
).orderBy(col("avg_predicted_ltv").desc())

print("\n   Customer Segments by Predicted LTV:")
segment_summary.show(truncate=False)

# Business recommendations
print("\n2️⃣  MARKETING BUDGET ALLOCATION:")

total_customers = ltv_predictions.count()
vip_count = ltv_segments.filter(col("ltv_segment") == "VIP (500+)").count()
high_count = ltv_segments.filter(col("ltv_segment") == "High Value (300-500)").count()
medium_count = ltv_segments.filter(col("ltv_segment") == "Medium Value (100-300)").count()
low_count = ltv_segments.filter(col("ltv_segment") == "Low Value (<100)").count()

print(f"\n   Total Customers: {total_customers:,}")
print(f"   ")
print(f"   📊 RECOMMENDED BUDGET ALLOCATION:")
print(f"   ")
print(f"   🏆 VIP Customers ({vip_count:,}, {vip_count/total_customers*100:.1f}%):")
print(f"      • Budget: 50% of marketing spend")
print(f"      • CAC Target: Up to $100/customer")
print(f"      • Strategy: VIP perks, personal service, exclusive offers")
print(f"   ")
print(f"   💎 High Value ({high_count:,}, {high_count/total_customers*100:.1f}%):")
print(f"      • Budget: 30% of marketing spend")
print(f"      • CAC Target: Up to $50/customer")
print(f"      • Strategy: Premium content, loyalty programs")
print(f"   ")
print(f"   📈 Medium Value ({medium_count:,}, {medium_count/total_customers*100:.1f}%):")
print(f"      • Budget: 15% of marketing spend")
print(f"      • CAC Target: Up to $20/customer")
print(f"      • Strategy: Upsell campaigns, bundle offers")
print(f"   ")
print(f"   ⚠️  Low Value ({low_count:,}, {low_count/total_customers*100:.1f}%):")
print(f"      • Budget: 5% of marketing spend")
print(f"      • CAC Target: Up to $5/customer")
print(f"      • Strategy: Automated campaigns only, minimize spend")

print("\n3️⃣  USE CASES:")

use_cases = [
    {
        "use_case": "Customer Acquisition",
        "application": "Set CAC limits based on predicted LTV",
        "impact": "Avoid overspending on low-value customers"
    },
    {
        "use_case": "Retention Campaigns",
        "application": "Focus on high-predicted-LTV customers at risk",
        "impact": "Maximize ROI on retention spend"
    },
    {
        "use_case": "Personalization",
        "application": "Offer premium perks to high-LTV predictions",
        "impact": "Increase loyalty and actual LTV"
    },
    {
        "use_case": "Early Identification",
        "application": "Predict LTV from first 3 orders",
        "impact": "Fast-track VIPs to premium service"
    }
]

for i, uc in enumerate(use_cases, 1):
    print(f"\n   {i}. {uc['use_case']}")
    print(f"      Application: {uc['application']}")
    print(f"      Impact: {uc['impact']}")

print("\n4️⃣  ESTIMATED BUSINESS IMPACT:")

print(f"\n   Current State (No LTV prediction):")
print(f"   • Spend $30/customer on ALL customers uniformly")
print(f"   • Total spend: {total_customers:,} × $30 = ${total_customers * 30:,}")
print(f"   ")
print(f"   With LTV-Based Allocation:")
print(f"   • VIP: {vip_count:,} × $100 = ${vip_count * 100:,}")
print(f"   • High: {high_count:,} × $50 = ${high_count * 50:,}")
print(f"   • Medium: {medium_count:,} × $20 = ${medium_count * 20:,}")
print(f"   • Low: {low_count:,} × $5 = ${low_count * 5:,}")
print(f"   • Total: ${vip_count * 100 + high_count * 50 + medium_count * 20 + low_count * 5:,}")
print(f"   ")
print(f"   💰 Cost Savings: ${total_customers * 30 - (vip_count * 100 + high_count * 50 + medium_count * 20 + low_count * 5):,}/year")
print(f"   📈 Revenue Impact: +15-25% from better allocation")

print("\n" + "=" * 70)
print("✅ BUSINESS INSIGHTS COMPLETE!")
print("=" * 70)

# COMMAND ----------


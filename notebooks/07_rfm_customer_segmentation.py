# Databricks notebook source
# Cell 1: RFM Customer Segmentation Setup

print("=" * 70)
print("📊 RFM CUSTOMER SEGMENTATION ANALYSIS")
print("=" * 70)

print("\n🎯 WHAT IS RFM?")
print("   RFM = Recency, Frequency, Monetary")
print("   A proven marketing framework to segment customers")
print("   ")
print("   • Recency (R):   How recently did they order?")
print("   • Frequency (F): How often do they order?")
print("   • Monetary (M):  How much do they buy?")

print("\n💡 WHY RFM MATTERS:")
print("   ✅ Used by Amazon, Walmart, Starbucks")
print("   ✅ Simple, interpretable, actionable")
print("   ✅ Helps target the right customers with right message")
print("   ✅ Improves ROI on marketing spend")

print("\n🎯 TODAY'S OBJECTIVES:")
print("   1. Analyze RFM scores we created in Gold layer")
print("   2. Profile each customer segment")
print("   3. Generate business insights & recommendations")
print("   4. Create actionable marketing strategies")

# Check data
print("\n🔍 CHECKING DATA:")
user_features = spark.table("gold_db.user_features")
user_count = user_features.count()

print(f"   ✅ User features table found")
print(f"   Total customers: {user_count:,}")

# Check RFM columns exist
rfm_cols = ['rfm_recency_score', 'rfm_frequency_score', 'rfm_monetary_score', 'rfm_segment']
missing = [col for col in rfm_cols if col not in user_features.columns]

if missing:
    print(f"   ⚠️  Missing columns: {missing}")
    print(f"   Need to create RFM scores first!")
else:
    print(f"   ✅ All RFM columns present")

print("\n" + "=" * 70)
print("✅ READY FOR RFM ANALYSIS!")
print("=" * 70)

# COMMAND ----------

# Cell 2: RFM Segment Distribution & Overview

print("=" * 70)
print("📊 RFM SEGMENT DISTRIBUTION")
print("=" * 70)

from pyspark.sql.functions import col, count, avg, sum as spark_sum, min as spark_min, max as spark_max

# Load user features
user_features = spark.table("gold_db.user_features")

# === SEGMENT DISTRIBUTION ===
print("\n1️⃣  CUSTOMER SEGMENT DISTRIBUTION:")

segment_dist = user_features.groupBy("rfm_segment").agg(
    count("*").alias("customer_count")
).orderBy(col("customer_count").desc())

total_customers = user_features.count()

# Show distribution
print(f"\n   Total Customers: {total_customers:,}")
print("\n   Segment Breakdown:")

segment_dist_pd = segment_dist.toPandas()
for _, row in segment_dist_pd.iterrows():
    segment = row['rfm_segment']
    count_val = row['customer_count']
    pct = (count_val / total_customers) * 100
    
    # Add emoji based on segment
    emoji = {
        'CHAMPIONS': '🏆',
        'LOYAL': '💎',
        'POTENTIAL': '🌟',
        'PROMISING': '🆕',
        'AT_RISK': '⚠️',
        'NEED_ATTENTION': '🔔',
        'HIBERNATING': '💤'
    }.get(segment, '📍')
    
    bar_length = int(pct / 2)  # Scale bar
    bar = '█' * bar_length
    
    print(f"   {emoji} {segment:15s} │ {count_val:>6,} ({pct:>5.1f}%) {bar}")

# === SEGMENT CHARACTERISTICS ===
print("\n2️⃣  SEGMENT CHARACTERISTICS:")

segment_profile = user_features.groupBy("rfm_segment").agg(
    count("*").alias("customers"),
    avg("rfm_recency_score").alias("avg_recency"),
    avg("rfm_frequency_score").alias("avg_frequency"),
    avg("rfm_monetary_score").alias("avg_monetary"),
    avg("total_orders").alias("avg_orders"),
    avg("avg_basket_size").alias("avg_basket"),
    avg("user_reorder_ratio").alias("avg_reorder_rate")
).orderBy(col("customers").desc())

print("\n   Profile by Segment:")
segment_profile.show(truncate=False)

# === RFM SCORE DISTRIBUTION ===
print("\n3️⃣  RFM SCORE DISTRIBUTION:")

print("\n   Recency Scores (1=Old, 5=Recent):")
user_features.groupBy("rfm_recency_score").count().orderBy("rfm_recency_score").show()

print("   Frequency Scores (1=Rare, 5=Frequent):")
user_features.groupBy("rfm_frequency_score").count().orderBy("rfm_frequency_score").show()

print("   Monetary Scores (1=Small, 5=Large baskets):")
user_features.groupBy("rfm_monetary_score").count().orderBy("rfm_monetary_score").show()

print("\n" + "=" * 70)
print("✅ SEGMENT OVERVIEW COMPLETE!")
print("=" * 70)

# COMMAND ----------

# Cell 3: Marketing Strategies by Segment

print("=" * 70)
print("📢 MARKETING STRATEGIES BY SEGMENT")
print("=" * 70)

strategies = {
    "CHAMPIONS": {
        "emoji": "🏆",
        "count": 928,
        "priority": "RETAIN (Critical)",
        "value": "Highest LTV - Protect at all costs",
        "actions": [
            "VIP loyalty program with exclusive benefits",
            "Early access to new products",
            "Personalized concierge service",
            "Thank-you gifts and surprise rewards",
            "Refer-a-friend program (they're advocates!)"
        ],
        "kpis": [
            "Retention rate > 95%",
            "Referrals per customer > 2",
            "NPS score > 70"
        ]
    },
    
    "AT_RISK": {
        "emoji": "⚠️",
        "count": 527,
        "priority": "SAVE (URGENT!)",
        "value": "Former Champions - High recovery value",
        "actions": [
            "Personal outreach: 'We miss you!' email",
            "Win-back offer: 25% off next order",
            "Survey: 'Why did you stop ordering?'",
            "Re-engagement campaign (3 touchpoints)",
            "If no response after 60 days: Stop marketing"
        ],
        "kpis": [
            "30-day reactivation rate > 20%",
            "Recovery to Loyal segment > 10%",
            "Survey response rate > 15%"
        ]
    },
    
    "LOYAL": {
        "emoji": "💎",
        "count": 9804,
        "priority": "UPSELL (High)",
        "value": "Frequent buyers - Increase basket size",
        "actions": [
            "'Complete your cart' recommendations",
            "Bundle deals: 'Buy 3 for the price of 2'",
            "Minimum order discounts: '$5 off orders $50+'",
            "Cross-sell related products",
            "Gamification: 'You're 3 items away from free shipping'"
        ],
        "kpis": [
            "Avg basket size increase: +20% (9.4 → 11.3 items)",
            "Bundle attach rate > 15%",
            "Free shipping threshold hit > 40%"
        ]
    },
    
    "PROMISING": {
        "emoji": "🆕",
        "count": 10324,
        "priority": "NURTURE (High)",
        "value": "New customers - Make or break phase",
        "actions": [
            "7-day welcome email series with tips",
            "First-purchase follow-up: 'How was it?'",
            "2nd order incentive: '10% off your next order'",
            "Product discovery: 'Others like you also bought...'",
            "Measure: Track 30-day conversion to Loyal"
        ],
        "kpis": [
            "30-day retention > 50%",
            "Conversion to Loyal > 25%",
            "Orders in first 30 days > 3"
        ]
    },
    
    "NEED_ATTENTION": {
        "emoji": "🔔",
        "count": 2720,
        "priority": "RE-ENGAGE (Medium)",
        "value": "Slipping away - Act before it's too late",
        "actions": [
            "Re-engagement email: 'Come back! Here's 15% off'",
            "Cart abandonment reminders",
            "Browse history recommendations",
            "Limited-time offers to create urgency",
            "A/B test different incentives"
        ],
        "kpis": [
            "60-day reactivation rate > 30%",
            "Email open rate > 25%",
            "Offer redemption > 10%"
        ]
    },
    
    "HIBERNATING": {
        "emoji": "💤",
        "count": 14047,
        "priority": "WIN-BACK or SUNSET (Low)",
        "value": "Inactive - Last-chance or give up",
        "actions": [
            "Final win-back: 'Last chance! 30% off'",
            "Exit survey: 'Why did you leave?'",
            "If no response after 90 days: Stop marketing (save costs)",
            "Suppression list: Don't waste ad spend",
            "Learn: Analyze why they churned"
        ],
        "kpis": [
            "90-day reactivation rate > 5%",
            "Survey completion > 10%",
            "Cost savings from suppression list"
        ]
    },
    
    "POTENTIAL": {
        "emoji": "🌟",
        "count": 5103,
        "priority": "GROW (Medium)",
        "value": "Large baskets but infrequent - Increase frequency",
        "actions": [
            "Subscription offers: 'Subscribe & save 10%'",
            "Reminder campaigns: 'Time to restock?'",
            "Push notifications: 'Your favorites are back'",
            "Auto-reorder suggestions",
            "Frequency incentives: 'Order 2x/month, get free shipping'"
        ],
        "kpis": [
            "Order frequency increase: +30%",
            "Subscription adoption > 15%",
            "Move to Loyal segment > 20%"
        ]
    }
}

# Print strategies
for segment, data in strategies.items():
    print(f"\n{data['emoji']} {segment}")
    print("=" * 70)
    print(f"   Customers:  {data['count']:,}")
    print(f"   Priority:   {data['priority']}")
    print(f"   Value:      {data['value']}")
    
    print(f"\n   📋 ACTION PLAN:")
    for i, action in enumerate(data['actions'], 1):
        print(f"      {i}. {action}")
    
    print(f"\n   📊 SUCCESS METRICS:")
    for kpi in data['kpis']:
        print(f"      • {kpi}")
    print()

print("=" * 70)
print("✅ MARKETING STRATEGIES COMPLETE!")
print("=" * 70)

# COMMAND ----------

# Cell 4: Executive Summary & Business Recommendations

print("=" * 70)
print("📊 RFM ANALYSIS - EXECUTIVE SUMMARY")
print("=" * 70)

# Load data
user_features = spark.table("gold_db.user_features")
total_customers = user_features.count()

print(f"\n📈 CUSTOMER BASE OVERVIEW:")
print(f"   Total Customers: {total_customers:,}")

# Calculate segment percentages
segment_counts = user_features.groupBy("rfm_segment").count().collect()
segment_dict = {row['rfm_segment']: row['count'] for row in segment_counts}

print(f"\n🎯 KEY FINDINGS:")

print(f"\n   1. 🚨 CUSTOMER HEALTH CONCERN:")
print(f"      • {segment_dict.get('HIBERNATING', 0):,} customers (32.3%) are HIBERNATING")
print(f"      • {segment_dict.get('AT_RISK', 0):,} customers (1.2%) are AT RISK")
print(f"      • {segment_dict.get('NEED_ATTENTION', 0):,} customers (6.3%) NEED ATTENTION")
print(f"      → Total at-risk: ~17,300 customers (40% of base!)")

print(f"\n   2. 💎 GROWTH OPPORTUNITY:")
print(f"      • Only {segment_dict.get('CHAMPIONS', 0):,} Champions (2.1%)")
print(f"      • Industry benchmark: 10-15% should be Champions")
print(f"      • Opportunity: Convert Loyal → Champions (increase basket size)")

print(f"\n   3. 🆕 NEW CUSTOMER CHALLENGE:")
print(f"      • {segment_dict.get('PROMISING', 0):,} Promising customers (23.8%)")
print(f"      • Critical phase: Will they become Loyal or Hibernating?")
print(f"      • Need: Better onboarding & nurturing")

print(f"\n   4. 💰 REVENUE CONCENTRATION:")
print(f"      • Top 2.1% (Champions) likely drive 30-40% of revenue")
print(f"      • 22.6% (Loyal) drive another 30-35%")
print(f"      • Risk: Over-reliance on small customer base")

print(f"\n💡 TOP 5 STRATEGIC PRIORITIES:")

priorities = [
    {
        "rank": 1,
        "priority": "SAVE AT-RISK CUSTOMERS (URGENT)",
        "impact": "High",
        "effort": "Medium",
        "details": [
            "527 former Champions stopped ordering",
            "Win-back campaign: 25% off + personal outreach",
            "Survey to understand why they left",
            "Expected recovery: 20% (105 customers)",
            "Timeline: Start immediately, 30-day campaign"
        ]
    },
    {
        "rank": 2,
        "priority": "NURTURE PROMISING CUSTOMERS",
        "impact": "Very High",
        "effort": "Medium",
        "details": [
            "10,324 new customers at make-or-break phase",
            "7-day welcome series + 2nd order incentive",
            "Goal: 25% conversion to Loyal (2,581 customers)",
            "Prevent: Churning to Hibernating (32% risk)",
            "Timeline: 30-day program, ongoing"
        ]
    },
    {
        "rank": 3,
        "priority": "UPSELL LOYAL CUSTOMERS",
        "impact": "High",
        "effort": "Low",
        "details": [
            "9,804 frequent buyers with small baskets (9.4 items)",
            "Bundle deals + minimum order discounts",
            "Goal: Increase basket to 11.3 items (+20%)",
            "Potential revenue lift: +20% from this segment",
            "Timeline: Ongoing optimization"
        ]
    },
    {
        "rank": 4,
        "priority": "PROTECT CHAMPIONS",
        "impact": "Critical",
        "effort": "Low",
        "details": [
            "928 VIP customers (75% reorder rate, 19.5 items/order)",
            "VIP program + exclusive perks + personal service",
            "Goal: 95%+ retention, prevent any churn",
            "Value: Each Champion worth 10x average customer",
            "Timeline: Immediate and ongoing"
        ]
    },
    {
        "rank": 5,
        "priority": "HIBERNATING CUSTOMER DECISION",
        "impact": "Medium",
        "effort": "Low",
        "details": [
            "14,047 inactive customers (32.3%)",
            "Final win-back attempt: 30% off",
            "After 90 days no response: Stop marketing",
            "Benefit: Reduce marketing costs by ~25%",
            "Timeline: 90-day sunset program"
        ]
    }
]

for p in priorities:
    print(f"\n   {p['rank']}. {p['priority']}")
    print(f"      Impact: {p['impact']} | Effort: {p['effort']}")
    for detail in p['details']:
        print(f"      • {detail}")

print(f"\n💰 ESTIMATED BUSINESS IMPACT (12 months):")

print(f"\n   Scenario: Implement all 5 priorities")
print(f"   ")
print(f"   Revenue Impact:")
print(f"   • Save 20% of At-Risk (105 customers × $1,000/year) = +$105K")
print(f"   • Convert 25% Promising → Loyal (2,581 × $500/year) = +$1.29M")
print(f"   • Upsell Loyal +20% (9,804 × $200 incremental) = +$1.96M")
print(f"   • Retain Champions 95% (prevent 46 churns × $2,000) = +$92K")
print(f"   ───────────────────────────────────────")
print(f"   TOTAL REVENUE IMPACT: +$3.45M/year")
print(f"   ")
print(f"   Cost Savings:")
print(f"   • Stop marketing to 14K Hibernating = -$70K/year marketing cost")
print(f"   ")
print(f"   NET IMPACT: +$3.52M/year")

print(f"\n🎯 IMPLEMENTATION ROADMAP:")

print(f"\n   MONTH 1 (URGENT):")
print(f"   ✅ Launch At-Risk win-back campaign")
print(f"   ✅ Start Promising customer nurture program")
print(f"   ✅ Implement Champion VIP perks")

print(f"\n   MONTH 2-3 (BUILD):")
print(f"   ✅ Roll out Loyal upsell campaigns")
print(f"   ✅ A/B test bundle strategies")
print(f"   ✅ Survey Hibernating customers")

print(f"\n   MONTH 4-6 (OPTIMIZE):")
print(f"   ✅ Analyze results, optimize campaigns")
print(f"   ✅ Begin Hibernating sunset process")
print(f"   ✅ Measure segment migrations")

print(f"\n   MONTH 7-12 (SCALE):")
print(f"   ✅ Scale successful campaigns")
print(f"   ✅ Continuous segment monitoring")
print(f"   ✅ Quarterly RFM refresh")

print(f"\n📊 SUCCESS METRICS (Track Monthly):")

metrics = [
    ("Segment Distribution", "Target: Champions 10%, Hibernating <20%"),
    ("At-Risk Recovery Rate", "Target: 20% reactivation"),
    ("Promising Conversion", "Target: 25% to Loyal in 30 days"),
    ("Loyal Basket Size", "Target: +20% (9.4 → 11.3 items)"),
    ("Champion Retention", "Target: 95%+"),
    ("Revenue per Segment", "Track monthly trends"),
    ("Customer Lifetime Value", "Track by segment")
]

for metric, target in metrics:
    print(f"   • {metric:30s}: {target}")

print("\n" + "=" * 70)
print("🎉 RFM ANALYSIS COMPLETE!")
print("=" * 70)

print(f"\n📝 NEXT STEPS:")
print(f"   1. Present findings to marketing team")
print(f"   2. Prioritize campaigns (start with At-Risk)")
print(f"   3. Set up tracking dashboards")
print(f"   4. Schedule monthly RFM refresh")
print(f"   5. Build Tableau dashboard for visualization")

# COMMAND ----------

# Cell 5: Export RFM Segments for Marketing Use

print("=" * 70)
print("💾 EXPORTING RFM SEGMENTS")
print("=" * 70)

# ADD THIS LINE AT THE TOP:
from pyspark.sql.functions import col, count, avg

# Create marketing-ready segment table
print("\n📊 Creating marketing segment table...")

marketing_segments = user_features.select(
    "user_id",
    "rfm_segment",
    "rfm_recency_score",
    "rfm_frequency_score",
    "rfm_monetary_score",
    "total_orders",
    "avg_basket_size",
    "user_reorder_ratio",
    "favorite_department",
    "preferred_order_time"
)

# Save as table
marketing_segments.write.mode("overwrite").saveAsTable("gold_db.customer_segments_marketing")

print(f"   ✅ Saved to: gold_db.customer_segments_marketing")
print(f"   Rows: {marketing_segments.count():,}")

# Show sample for each segment
print(f"\n📋 SAMPLE CUSTOMERS BY SEGMENT:")

for segment in ['CHAMPIONS', 'AT_RISK', 'LOYAL', 'PROMISING', 'HIBERNATING']:
    print(f"\n   {segment}:")
    marketing_segments.filter(col("rfm_segment") == segment).show(3, truncate=False)

# Create summary stats by segment
print(f"\n📊 SEGMENT SUMMARY STATS:")

segment_summary = marketing_segments.groupBy("rfm_segment").agg(
    count("*").alias("customers"),
    avg("total_orders").alias("avg_orders"),
    avg("avg_basket_size").alias("avg_basket"),
    avg("user_reorder_ratio").alias("avg_reorder_rate")
).orderBy(col("customers").desc())

segment_summary.show(truncate=False)

print("\n✅ Segments ready for marketing team!")
print("   Table: gold_db.customer_segments_marketing")

print("\n" + "=" * 70)
print("✅ RFM EXPORT COMPLETE!")
print("=" * 70)
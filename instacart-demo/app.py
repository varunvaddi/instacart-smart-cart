import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random
import math
from pathlib import Path
# ============================================================================
# DRIFT CALCULATION FUNCTION (FOR MONITORING TAB)
# ============================================================================
def calculate_data_drift(df1, df2, feature='avg_basket_size'):
    """
    Calculate KL divergence (data drift) between two distributions
    """
    from scipy.stats import entropy
    import numpy as np
    
    # Create histograms
    hist1, bins = np.histogram(df1[feature].dropna(), bins=20, density=True)
    hist2, _ = np.histogram(df2[feature].dropna(), bins=bins, density=True)
    
    # Add small constant to avoid log(0)
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    # Calculate KL divergence
    kl_div = entropy(hist1, hist2)
    
    return kl_div

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Instacart ML Portfolio",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================
BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    try:
        users = pd.read_csv(BASE_DIR / "sample_users.csv")
        reorder_recs = pd.read_csv(BASE_DIR / "reorder_recommendations.csv")
        rules = pd.read_csv(BASE_DIR / "association_rules.csv")
        metrics = pd.read_csv(BASE_DIR / "model_metrics.csv")
        return users, reorder_recs, rules, metrics
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

users, reorder_recs, rules, metrics = load_data()

# ============================================================================
# HEADER
# ============================================================================
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown('<h1 class="main-header">🛒 Instacart Smart Cart</h1>', unsafe_allow_html=True)
    st.markdown("**ML-Powered Recommendation Engine | End-to-End Portfolio**")

with col2:
    st.markdown("### Links")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/varunvaddi)")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://linkedin.com/in/varunvaddi)")

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=80)
st.sidebar.title("🎯 Customer Selector")

user_id = st.sidebar.selectbox(
    "Choose Customer:",
    sorted(users['user_id'].unique()),
    format_func=lambda x: f"User {x} ({users[users['user_id']==x]['rfm_segment'].values[0]})"
)

user = users[users['user_id'] == user_id].iloc[0]

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Profile")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Segment", user['rfm_segment'])
    st.metric("Orders", f"{int(user['total_orders'])}")
with col2:
    st.metric("Basket", f"{user['avg_basket_size']:.1f}")
    st.metric("Reorder", f"{user['user_reorder_ratio']:.0%}")

st.sidebar.markdown("### 📊 RFM Scores")
for metric, score in [
    ('Recency', int(user['rfm_recency_score'])),
    ('Frequency', int(user['rfm_frequency_score'])),
    ('Monetary', int(user['rfm_monetary_score']))
]:
    st.sidebar.progress(score / 5, text=f"{metric}: {score}/5")

# ============================================================================
# MAIN TABS - REORGANIZED!
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🛒 Recommendations",
    "🧪 What-If Simulator",
    "💰 ROI Calculator",
    "🧪 A/B Test Calculator",
    "🔔 Monitoring",
    "📊 Analytics & Models",
    "💡 About"
])

# ============================================================================
# TAB 1: RECOMMENDATIONS (Original + Email Preview)
# ============================================================================
with tab1:
    st.header("🎯 Personalized Recommendations")
    st.markdown(f"**AI-powered suggestions for User {user_id} ({user['rfm_segment']} segment)**")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # ========== BUY AGAIN ==========
        st.markdown("### 📦 Buy Again")
        st.caption("ML model predictions (80% AUC)")
        
        user_recs = reorder_recs[reorder_recs['user_id'] == user_id].head(10)
        
        if len(user_recs) > 0:
            for idx, rec in user_recs.iterrows():
                prob = 0.75
                if 'probability' in rec and pd.notna(rec['probability']):
                    try:
                        prob_str = str(rec['probability'])
                        if '[' in prob_str:
                            prob = float(prob_str.split(',')[1].replace(']', ''))
                    except:
                        pass
                
                col_prod, col_prob = st.columns([3, 1])
                with col_prod:
                    st.markdown(f"**{rec['product_name']}**")
                with col_prob:
                    st.markdown(f"`{prob:.0%}`")
                st.progress(prob)
                st.markdown("")
        else:
            st.info("📭 No predictions available")
        
        st.markdown("---")
        
        # ========== FREQUENTLY BOUGHT TOGETHER ==========
        st.markdown("### 🔗 Frequently Bought Together")
        st.caption("Association rules from market basket analysis")
        
        top_rules = rules.nlargest(8, 'lift')
        
        for idx, rule in top_rules.iterrows():
            st.markdown(f"**{rule['product_a'][:35]}** → **{rule['product_b'][:35]}**")
            
            col_conf, col_lift = st.columns(2)
            with col_conf:
                st.caption(f"✓ {rule['confidence_a_to_b']:.0%} buy together")
            with col_lift:
                st.caption(f"⚡ {rule['lift']:.1f}x lift")
            st.markdown("")
    
    with col2:
        # ========== SEGMENT RECOMMENDATIONS ==========
        st.markdown(f"### 🎯 Popular in {user['rfm_segment']}")
        st.caption("What similar customers love")
        
        segment_products = {
            'CHAMPIONS': ['🏆 Organic Avocado', '🥗 Baby Spinach', '🍓 Strawberries', '🍋 Lemon', '🥛 Organic Milk'],
            'LOYAL': ['🍌 Banana', '🥑 Avocado', '🍓 Berries', '🥬 Spinach', '🍊 Oranges'],
            'PROMISING': ['🍌 Banana', '🍓 Strawberries', '🥑 Avocado', '🥬 Spinach', '🍋 Lemon'],
            'HIBERNATING': ['💤 Win-back offers', '🎁 Special discounts', '📧 Reminder emails'],
            'AT_RISK': ['🚨 Urgent retention', '💰 Deep discounts', '🎁 Free shipping'],
            'POTENTIAL': ['🍌 Staples', '🥑 Popular items', '🍓 Fresh produce'],
            'NEED_ATTENTION': ['🔔 Re-engagement', '🍌 Favorites', '💝 Loyalty rewards']
        }
        
        products = segment_products.get(user['rfm_segment'], ['🍌 Banana'])
        for i, p in enumerate(products, 1):
            st.markdown(f"{i}. {p}")
        
        st.markdown("---")
        
        # ========== CUSTOMER INSIGHTS ==========
        st.markdown("### 💡 Insights")
        
        engagement = (user['rfm_recency_score'] + user['rfm_frequency_score'] + user['rfm_monetary_score']) / 15
        churn_risk = 1 - engagement
        
        st.metric("Engagement", f"{engagement:.0%}")
        
        if churn_risk > 0.6:
            st.error(f"⚠️ High churn risk: {churn_risk:.0%}")
        elif churn_risk > 0.3:
            st.warning(f"⚡ Medium risk: {churn_risk:.0%}")
        else:
            st.success(f"✅ Low risk: {churn_risk:.0%}")
        
        segment_actions = {
            'CHAMPIONS': '🏆 VIP perks & exclusive access',
            'LOYAL': '💎 Upsell bundles',
            'PROMISING': '🆕 Onboarding series',
            'HIBERNATING': '💤 Win-back (30% off)',
            'AT_RISK': '🚨 Urgent retention (25% off)',
            'POTENTIAL': '🌟 Increase frequency',
            'NEED_ATTENTION': '🔔 Re-engagement'
        }
        
        st.info(f"**Action:** {segment_actions.get(user['rfm_segment'], 'Standard marketing')}")
    
    # ========== EMAIL PREVIEW (Native Streamlit) ==========
    st.markdown("---")
    st.markdown("### 📧 AI-Generated Marketing Email Preview")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        email_type = st.selectbox(
            "Email Campaign:",
            ["Personalized Recommendations", "Churn Prevention", "Bundle Offer", "Win-Back"]
        )
        
        send_time = st.selectbox("Send Time:", ["Morning (8-10 AM)", "Lunch (12-2 PM)", "Evening (6-8 PM)"])
        
        st.markdown("---")
        st.info("""
        **Integration:**
        • Azure Logic Apps
        • SendGrid
        • Databricks webhooks
        """)
    
    with col2:
        if email_type == "Personalized Recommendations":
            subject = f"🛒 Your Weekly Picks"
            
            st.markdown(f"**📬 Subject:** {subject}")
            st.markdown(f"**⏰ Send:** {send_time}")
            st.markdown("---")
            
            st.markdown("## 🛒 Personalized Just For You")
            st.markdown(f"Hi! Based on your **{int(user['total_orders'])} orders:**")
            
            st.markdown("### 📦 Ready to Reorder")
            
            for idx, rec in user_recs.head(3).iterrows():
                st.markdown(f"✓ **{rec['product_name']}**")
            
            st.markdown("""
            <p style='text-align: center; margin: 20px 0;'>
                <a href='#' style='background-color: #FF6B6B; color: white; padding: 15px 30px; 
                   text-decoration: none; border-radius: 5px; font-weight: bold;'>
                    🛒 Shop Now
                </a>
            </p>
            """, unsafe_allow_html=True)
            
            st.caption("Generated by ML • Reorder model (80% AUC)")
        
        elif email_type == "Churn Prevention":
            subject = "⚠️ We Miss You! 25% Off"
            
            st.markdown(f"**📬 Subject:** {subject}")
            st.markdown(f"**⏰ Send:** {send_time}")
            st.markdown("---")
            
            st.warning("### ⚠️ We Haven't Seen You!")
            st.markdown("Get **25% OFF** with code: `COMEBACK25`")
            
            st.markdown("**Your Favorites:**")
            st.markdown("✓ Organic Bananas")
            st.markdown("✓ Baby Spinach")
            st.markdown("✓ Strawberries")
            
            st.markdown("""
            <p style='text-align: center; margin: 20px 0;'>
                <a href='#' style='background-color: #856404; color: white; padding: 15px 30px; 
                   text-decoration: none; border-radius: 5px; font-weight: bold;'>
                    🎁 Claim 25% Off
                </a>
            </p>
            """, unsafe_allow_html=True)
            
            st.caption("Churn model (77% AUC) • Expires in 7 days")
        
        elif email_type == "Bundle Offer":
            subject = "🎁 Bundle - 15% OFF"
            
            st.markdown(f"**📬 Subject:** {subject}")
            st.markdown(f"**⏰ Send:** {send_time}")
            st.markdown("---")
            
            st.info("### 🎁 Special Bundle")
            
            st.markdown("**Includes:**")
            st.markdown("✓ Greek Yogurt (3 pack)")
            st.markdown("✓ Fresh Berries (2 pints)")
            st.markdown("✓ Granola + Honey")
            
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown("~~$24.99~~")
            with col_b:
                st.markdown("### $21.24")
            
            st.caption("62% who buy yogurt also buy berries")
            
            st.markdown("""
            <p style='text-align: center; margin: 20px 0;'>
                <a href='#' style='background-color: #0c5460; color: white; padding: 15px 30px; 
                   text-decoration: none; border-radius: 5px; font-weight: bold;'>
                    🛒 Add Bundle
                </a>
            </p>
            """, unsafe_allow_html=True)
        
        else:  # Win-Back
            subject = "🚨 Last Chance: 30% OFF"
            
            st.markdown(f"**📬 Subject:** {subject}")
            st.markdown(f"**⏰ Send:** {send_time}")
            st.markdown("---")
            
            st.error("### 🚨 Final Offer")
            
            st.markdown("""
            <div style='text-align: center; padding: 20px; background-color: white; border-radius: 10px;'>
                <h1 style='font-size: 3rem; color: #721c24;'>30% OFF</h1>
                <h2 style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>WINBACK30</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Plus:** Free delivery over $35")
            
            st.markdown("""
            <p style='text-align: center; margin: 20px 0;'>
                <a href='#' style='background-color: #721c24; color: white; padding: 20px 40px; 
                   text-decoration: none; border-radius: 5px; font-weight: bold;'>
                    🎁 Claim Now
                </a>
            </p>
            """, unsafe_allow_html=True)
            
            st.error("⏰ Expires in 48 hours")

# ============================================================================
# TAB 2: WHAT-IF SIMULATOR (USING REAL ML LOGIC)
# ============================================================================
with tab2:
    st.header("🧪 What-If Simulator")
    st.markdown("**Adjust customer profile → See ML predictions change in real-time**")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📝 Customer Profile")
        
        sim_recency = st.slider("Recency Score", 1, 5, 3, help="How recently they ordered (5=recent)")
        sim_frequency = st.slider("Frequency Score", 1, 5, 3, help="How often they order (5=frequent)")
        sim_monetary = st.slider("Monetary Score", 1, 5, 3, help="How much they spend (5=high)")
        sim_basket = st.slider("Avg Basket Size", 1, 30, 10, help="Items per order")
        sim_reorder = st.slider("Reorder Ratio (%)", 0, 100, 50) / 100
        sim_days_between = st.slider("Days Between Orders", 1, 60, 15)
        
        # Calculate segment (using real logic)
        rfm_total = sim_recency + sim_frequency + sim_monetary
        
        if rfm_total >= 13 and sim_recency >= 4:
            pred_segment = "CHAMPIONS 🏆"
            color = "green"
        elif rfm_total >= 10 and sim_frequency >= 3:
            pred_segment = "LOYAL 💎"
            color = "blue"
        elif rfm_total >= 7 and sim_recency >= 4:
            pred_segment = "PROMISING 🆕"
            color = "orange"
        elif sim_recency <= 2 and sim_frequency >= 4:
            pred_segment = "AT RISK ⚠️"
            color = "red"
        elif sim_recency <= 2:
            pred_segment = "HIBERNATING 💤"
            color = "gray"
        elif sim_frequency <= 2:
            pred_segment = "NEED ATTENTION 🔔"
            color = "orange"
        else:
            pred_segment = "POTENTIAL 🌟"
            color = "blue"
        
        st.markdown(f"### Segment:")
        st.markdown(f"## :{color}[{pred_segment}]")
    
    with col2:
        st.markdown("### 🎯 ML Model Predictions")
        
        # ===== REAL CHURN MODEL LOGIC =====
        # Based on actual model: churn if (total_orders < 10) OR (recency_days > expected * 2)
        
        # Estimate total orders from frequency
        estimated_total_orders = sim_frequency * 8  # Frequency score * avg multiplier
        
        # Expected days based on frequency
        expected_days = 30 / max(sim_frequency, 1)  # Higher frequency = shorter expected gap
        
        # Churn logic (matching actual model)
        is_low_engagement = estimated_total_orders < 10
        is_inactive = sim_days_between > (expected_days * 2)
        
        if is_low_engagement or is_inactive:
            churn_risk = 0.75  # High risk
            if is_low_engagement and is_inactive:
                churn_risk = 0.85  # Very high risk
        else:
            # Calculate based on feature patterns
            engagement = (sim_recency + sim_frequency + sim_monetary) / 15
            churn_risk = max(0.1, 1 - engagement)  # Minimum 10% risk
        
        # Adjust based on reorder ratio (from model insights)
        if sim_reorder < 0.3:
            churn_risk = min(0.9, churn_risk + 0.15)  # Low reorders increase risk
        elif sim_reorder > 0.6:
            churn_risk = max(0.1, churn_risk - 0.10)  # High reorders decrease risk
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_risk * 100,
            title={'text': "Churn Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if churn_risk > 0.6 else "orange" if churn_risk > 0.3 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # ===== REAL LTV MODEL LOGIC =====
        # Based on actual model: LTV = total_orders × avg_basket_size
        # Use RFM to estimate
        
        # Estimate orders per year from frequency
        orders_per_year = sim_frequency * 2.5  # Frequency score to annual orders
        
        # LTV calculation (matching actual model pattern)
        annual_value = sim_basket * orders_per_year
        
        # Apply reorder multiplier (loyal customers = higher LTV)
        loyalty_multiplier = 1 + (sim_reorder * 0.5)  # Up to 1.5x for high reorders
        
        ltv_estimate = annual_value * loyalty_multiplier * 3  # 3-year LTV
        
        # Show predictions
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Predicted LTV", f"${ltv_estimate:.0f}", help="3-year customer value")
        with col_b:
            engagement = (sim_recency + sim_frequency + sim_monetary) / 15
            st.metric("Engagement", f"{engagement:.0%}")
        
        st.markdown("---")
        
        # ===== EXPLAIN THE PREDICTION =====
        st.markdown("### 💡 Prediction Explanation:")
        
        st.markdown("**🔍 Churn Risk Factors:**")
        
        if is_low_engagement:
            st.warning(f"⚠️ Low engagement: ~{estimated_total_orders:.0f} estimated orders (< 10 threshold)")
        
        if is_inactive:
            st.warning(f"⚠️ Inactive: {sim_days_between} days gap > {expected_days*2:.0f} expected")
        
        if sim_reorder < 0.3:
            st.warning(f"⚠️ Low reorder rate: {sim_reorder:.0%} (< 30% threshold)")
        
        if not (is_low_engagement or is_inactive) and sim_reorder >= 0.3:
            st.success("✅ Strong engagement patterns, low churn risk")
        
        st.markdown(f"**💰 LTV Calculation:**")
        st.info(
            f"""
        **Orders/year:** {orders_per_year:.1f}  
        
        **Basket size:** ${sim_basket:.0f}  
        
        **Loyalty multiplier:** {loyalty_multiplier:.2f}×  
        
        **3-year LTV:** ${ltv_estimate:.0f}
        """
        )

        st.markdown("---")
        st.markdown("### 🎯 Recommended Actions:")
        
        if churn_risk > 0.6:
            st.error("🚨 **HIGH RISK**: Immediate intervention needed")
            st.markdown("- Send 25% discount offer within 48 hours")
            st.markdown("- Personal outreach from account manager")
            st.markdown("- Add to high-priority retention list")
        elif churn_risk > 0.3:
            st.warning("⚡ **MEDIUM RISK**: Re-engagement campaign")
            st.markdown("- Trigger automated re-engagement email (15% off)")
            st.markdown("- Show personalized product recommendations")
            st.markdown("- Send reminder about favorite products")
        else:
            st.success("✅ **LOW RISK**: Focus on upselling")
            st.markdown("- Recommend premium products")
            st.markdown("- Offer bundle deals to increase basket size")
            st.markdown("- Enroll in loyalty program")
        
        # Show which model features matter
        st.markdown("---")
        st.caption("**Model Features Used:** RFM scores, basket size, reorder ratio, days between orders")
        st.caption("**Based on:** Churn model (77% AUC) + LTV model (90% R²)")

# ============================================================================
# TAB 3: ROI CALCULATOR
# ============================================================================
with tab3:
    st.header("💰 Interactive ROI Calculator")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📊 Assumptions")
        
        total_customers = st.number_input("Total Customers", 10000, 1000000, 100000, 10000)
        avg_order_value = st.slider("Avg Order ($)", 20, 200, 100, 5)
        orders_per_year = st.slider("Orders/Year", 2, 50, 12)
        
        st.markdown("---")
        
        churn_rate = st.slider("Current Churn (%)", 10, 50, 30) / 100
        churn_reduction = st.slider("ML Reduction (%)", 5, 30, 15) / 100
        
        st.markdown("---")
        
        rec_ctr = st.slider("Rec CTR (%)", 5, 30, 12) / 100
        rec_conv = st.slider("Rec Conversion (%)", 10, 50, 25) / 100
        
        st.markdown("---")
        
        ml_cost = st.number_input("ML Cost ($)", 10000, 500000, 100000, 10000)
    
    with col2:
        st.markdown("#### 💵 Results")
        
        annual_revenue = total_customers * avg_order_value * orders_per_year
        customers_churning = int(total_customers * churn_rate)
        
        st.metric("Current Revenue", f"${annual_revenue:,.0f}")
        st.metric("Churning", f"{customers_churning:,}")
        
        st.markdown("---")
        
        customers_saved = int(customers_churning * churn_reduction)
        churn_value = customers_saved * avg_order_value * orders_per_year
        
        recs_clicked = int(total_customers * rec_ctr)
        rec_revenue = int(recs_clicked * rec_conv * avg_order_value)
        
        total_value = churn_value + rec_revenue
        
        st.metric("Saved", f"{customers_saved:,}")
        st.metric("Churn Value", f"${churn_value:,.0f}")
        st.metric("Rec Revenue", f"${rec_revenue:,.0f}")
        
        st.markdown("---")
        st.markdown("### 🎯 Bottom Line")
        
        net_value = total_value - ml_cost
        roi = (net_value / ml_cost) * 100
        
        st.metric("Total Value", f"${total_value:,.0f}")
        st.metric("Cost", f"${ml_cost:,.0f}")
        st.metric("Net Benefit", f"${net_value:,.0f}")
        st.metric("ROI", f"{roi:.0f}%")
        
        if roi > 0:
            months = (ml_cost / total_value) * 12
            st.success(f"✅ Payback: {months:.1f} months")
        else:
            st.error("❌ Negative ROI")
    
    st.markdown("---")
    
    if roi > 200:
        st.success(f"🎉 Outstanding! ${roi/100:.2f} per dollar")
    elif roi > 100:
        st.info(f"📈 Good ROI: ${roi/100:.2f} per dollar")
    elif roi > 0:
        st.warning(f"⚠️ Modest: ${roi/100:.2f} per dollar")
    else:
        st.error("❌ Not profitable")

# ============================================================================
# TAB 4: A/B TEST CALCULATOR (ENHANCED WITH VISUALS)
# ============================================================================
with tab4:
    st.header("🧪 A/B Test Calculator")
    st.markdown("**Design statistically valid experiments**")
    
    # Quick explanation
    with st.expander("📖 What is A/B Testing?"):
        st.markdown("""
        **A/B testing** compares two versions to see which performs better.
        
        **Example:**
        - **Control (A):** Random recommendations → 3% conversion
        - **Treatment (B):** ML recommendations → 3.6% conversion
        - **Question:** Is the 20% improvement real or just luck?
        
        **How it works:**
        1. Split users randomly into 2 groups
        2. Show version A to Group 1, version B to Group 2
        3. Measure conversion rates
        4. Use statistics to determine if difference is significant
        """)
    
    st.markdown("---")
    
    # Calculator
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Test Setup")
        baseline = st.number_input(
            "Baseline Conversion (%)", 
            min_value=1.0, 
            max_value=10.0, 
            value=3.0,
            step=0.1,
            help="Current conversion rate (Control group)"
        ) / 100
        
        lift = st.number_input(
            "Target Lift (%)", 
            min_value=5, 
            max_value=50, 
            value=20,
            step=5,
            help="Minimum improvement you want to detect"
        )
        
        treatment_rate = baseline * (1 + lift/100)
        
        st.metric(
            "Treatment Rate", 
            f"{treatment_rate:.2%}",
            f"+{lift}%",
            help="Expected conversion with ML"
        )
    
    with col2:
        st.markdown("### ⚙️ Statistical Settings")
        
        confidence = st.selectbox(
            "Confidence Level", 
            [90, 95, 99], 
            index=1,
            help="How sure you want to be (95% = industry standard)"
        )
        
        power = st.selectbox(
            "Statistical Power", 
            [80, 90, 95], 
            index=1,
            help="Probability of detecting the lift (80% = standard)"
        )
        
        daily_traffic = st.number_input(
            "Daily Users",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100,
            help="How many users visit per day"
        )
    
    with col3:
        st.markdown("### 📈 Required Sample")
        
        # Calculate sample size
        z_alpha = 1.96 if confidence == 95 else (1.645 if confidence == 90 else 2.576)
        z_beta = 0.84 if power == 80 else (1.28 if power == 90 else 1.645)
        
        p1 = baseline
        p2 = treatment_rate
        
        n = 2 * ((z_alpha + z_beta)**2) * (p1*(1-p1) + p2*(1-p2)) / ((p2-p1)**2)
        n = math.ceil(n)
        
        st.metric(
            "Per Group", 
            f"{n:,}",
            help="Users needed in EACH group"
        )
        
        st.metric(
            "Total Users", 
            f"{n*2:,}",
            help="Total users needed for test"
        )
        
        test_days = math.ceil((n * 2) / daily_traffic)
        
        st.metric(
            "Test Duration", 
            f"{test_days} days",
            help="How long to run the test"
        )
    
    st.markdown("---")
    
    # Visual explanation
    st.markdown("### 📊 How A/B Testing Works")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 1️⃣ Split Traffic")
        
        # Create split visualization
        split_data = pd.DataFrame({
            'Group': ['Control (A)', 'Treatment (B)'],
            'Users': [n, n],
            'Version': ['Random Recs', 'ML Recs']
        })
        
        fig_split = px.bar(
            split_data,
            x='Group',
            y='Users',
            color='Group',
            text='Users',
            title=f"Traffic Split: {n:,} users each",
            color_discrete_map={
                'Control (A)': '#FF6B6B',
                'Treatment (B)': '#4ECDC4'
            }
        )
        fig_split.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_split.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_split, use_container_width=True)
        
        st.caption(f"Each group needs **{n:,} users** to detect {lift}% lift")
    
    with col2:
        st.markdown("#### 2️⃣ Measure Results")
        
        # Create results comparison
        results_data = pd.DataFrame({
            'Group': ['Control', 'Treatment'],
            'Conversion Rate': [baseline * 100, treatment_rate * 100],
            'Conversions': [int(n * baseline), int(n * treatment_rate)]
        })
        
        fig_results = go.Figure()
        
        fig_results.add_trace(go.Bar(
            name='Control',
            x=['Conversion Rate'],
            y=[baseline * 100],
            marker_color='#FF6B6B',
            text=[f'{baseline:.1%}'],
            textposition='outside'
        ))
        
        fig_results.add_trace(go.Bar(
            name='Treatment',
            x=['Conversion Rate'],
            y=[treatment_rate * 100],
            marker_color='#4ECDC4',
            text=[f'{treatment_rate:.1%}'],
            textposition='outside'
        ))
        
        fig_results.update_layout(
            title=f"Expected Results (+{lift}% lift)",
            yaxis_title='Conversion Rate (%)',
            height=300,
            barmode='group'
        )
        
        st.plotly_chart(fig_results, use_container_width=True)
        
        conversions_gained = int(n * treatment_rate) - int(n * baseline)
        st.caption(f"**+{conversions_gained:,}** additional conversions with ML")
    
    st.markdown("---")
    
    # Timeline visualization
    st.markdown("### 📅 Test Timeline")
    
    timeline_data = pd.DataFrame({
        'Day': list(range(1, test_days + 1)),
        'Control Users': [daily_traffic // 2] * test_days,
        'Treatment Users': [daily_traffic // 2] * test_days
    })
    
    timeline_data['Control Cumulative'] = timeline_data['Control Users'].cumsum()
    timeline_data['Treatment Cumulative'] = timeline_data['Treatment Users'].cumsum()
    
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Scatter(
        x=timeline_data['Day'],
        y=timeline_data['Control Cumulative'],
        mode='lines+markers',
        name='Control',
        line=dict(color='#FF6B6B', width=3),
        fill='tonexty'
    ))
    
    fig_timeline.add_trace(go.Scatter(
        x=timeline_data['Day'],
        y=timeline_data['Treatment Cumulative'],
        mode='lines+markers',
        name='Treatment',
        line=dict(color='#4ECDC4', width=3),
        fill='tozeroy'
    ))
    
    # Add target line
    fig_timeline.add_hline(
        y=n,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Target: {n:,} per group"
    )
    
    fig_timeline.update_layout(
        title=f"User Accumulation Over {test_days} Days",
        xaxis_title='Day',
        yaxis_title='Cumulative Users',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    st.markdown("---")
    
    # Business impact
    st.markdown("### 💰 Business Impact Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Assumptions:**")
        
        avg_order = st.slider(
            "Avg Order Value ($)",
            min_value=20,
            max_value=200,
            value=100,
            step=10
        )
        
        monthly_users = st.number_input(
            "Monthly Users",
            min_value=1000,
            max_value=1000000,
            value=30000,
            step=1000
        )
    
    with col2:
        st.markdown("**If ML Wins:**")
        
        # Current state
        current_conversions = int(monthly_users * baseline)
        current_revenue = current_conversions * avg_order
        
        # With ML
        ml_conversions = int(monthly_users * treatment_rate)
        ml_revenue = ml_conversions * avg_order
        
        # Lift
        additional_conversions = ml_conversions - current_conversions
        additional_revenue = ml_revenue - current_revenue
        
        st.metric(
            "Additional Conversions/Month",
            f"{additional_conversions:,}",
            f"+{lift}%"
        )
        
        st.metric(
            "Additional Revenue/Month",
            f"${additional_revenue:,}",
            f"+{lift}%"
        )
        
        st.metric(
            "Annual Impact",
            f"${additional_revenue * 12:,}",
            "if sustained"
        )
    
    st.markdown("---")
    
    # Statistical explanation
    st.markdown("### 📐 Statistical Concepts")
    
    tab_a, tab_b, tab_c = st.tabs(["Confidence Level", "Statistical Power", "Sample Size"])
    
    with tab_a:
        st.markdown("""
        ### What is Confidence Level?
        
        **Simple:** How sure you want to be that the difference is real.
        
        **95% Confidence = Industry Standard**
        - If you ran this test 100 times
        - 95 times you'd correctly identify the winner
        - 5 times you'd be wrong (false positive)
        
        **Example:**
        - 90% confidence = More false alarms, but faster test
        - 99% confidence = Very few false alarms, but longer test
        
        **Choose 95%** unless you have a specific reason!
        """)
    
    with tab_b:
        st.markdown("""
        ### What is Statistical Power?
        
        **Simple:** Probability of detecting the improvement if it exists.
        
        **80% Power = Industry Standard**
        - If ML really is better
        - 80% chance you'll detect it
        - 20% chance you'll miss it (false negative)
        
        **Example:**
        - 80% power = Standard, cost-effective
        - 90% power = More reliable, but needs more users
        
        **Choose 80%** for most tests!
        """)
    
    with tab_c:
        st.markdown(f"""
        ### Why {n:,} Users Per Group?
        
        **Formula:**
```
        n = 2 × (Zα + Zβ)² × (p₁(1-p₁) + p₂(1-p₂))
            ──────────────────────────────────────
                        (p₂ - p₁)²
```
        
        **Your Test:**
        - Baseline: {baseline:.1%}
        - Target: {treatment_rate:.1%} (+{lift}%)
        - Confidence: {confidence}%
        - Power: {power}%
        
        **Result:** Need {n:,} users per group
        
        **Why?**
        - Smaller lifts = Need more users
        - Higher confidence = Need more users
        - Higher power = Need more users
        
        **Rule of thumb:** 
        Detecting a {lift}% lift requires ~{n:,} users per group
        """)
    
    st.markdown("---")
    
    # Real example
    st.markdown("### 🎯 Real-World Example")
    
    st.info("""
    **Instacart ML Recommendations A/B Test**
    
    **Setup:**
    - **Control:** Random product recommendations
    - **Treatment:** ML-powered recommendations (from our models)
    - **Metric:** Click-through rate on recommendations
    - **Hypothesis:** ML recs will increase CTR by 20%
    
    **Test Design:**
    - Baseline CTR: 3% (historical data)
    - Target lift: 20% (to 3.6%)
    - Confidence: 95%
    - Power: 80%
    - Required: {} users per group
    - Duration: {} days (with {} daily users)
    
    **If Successful:**
    - {} additional conversions/month
    - ${:,} additional revenue/month
    - ${:,} annual impact
    
    **Decision Rule:**
    - If p-value < 0.05 → Launch ML recs to 100%
    - If p-value > 0.05 → Keep random recs, iterate on model
    """.format(
        n, test_days, daily_traffic,
        additional_conversions,
        additional_revenue,
        additional_revenue * 12
    ))
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Test Plan", use_container_width=True):
            test_plan = f"""
# A/B Test Plan: ML Recommendations

## Test Setup
- **Control:** Random recommendations
- **Treatment:** ML-powered recommendations
- **Primary Metric:** Conversion rate
- **Baseline:** {baseline:.2%}
- **Target:** {treatment_rate:.2%} (+{lift}%)

## Statistical Design
- **Confidence Level:** {confidence}%
- **Statistical Power:** {power}%
- **Sample Size:** {n:,} per group ({n*2:,} total)
- **Duration:** {test_days} days
- **Daily Traffic:** {daily_traffic:,} users

## Success Criteria
- p-value < 0.05
- Conversion rate lift ≥ {lift}%
- No degradation in order value or retention

## Business Impact (if successful)
- Monthly revenue: +${additional_revenue:,}
- Annual revenue: +${additional_revenue * 12:,}
- Additional conversions: {additional_conversions:,}/month

## Timeline
Day 0: Launch test
Day {test_days//2}: Mid-test check
Day {test_days}: Final analysis & decision

Generated: {datetime.now().strftime('%Y-%m-%d')}
"""
            
            st.download_button(
                "📄 Download",
                test_plan.encode(),
                f"ab_test_plan_{datetime.now().strftime('%Y%m%d')}.md",
                "text/markdown"
            )
    
    with col2:
        if st.button("📊 Simulate Results", use_container_width=True):
            st.info("Feature coming soon! Will simulate test outcomes with different scenarios.")
    
    with col3:
        if st.button("📚 Learn More", use_container_width=True):
            st.markdown("""
            **Resources:**
            - [Evan Miller's A/B Test Calculator](https://www.evanmiller.org/ab-testing/)
            - [Optimizely Stats Engine](https://www.optimizely.com/stats-engine/)
            - [Google's A/B Testing Guide](https://research.google/pubs/pub36500/)
            """)

# ============================================================================
# TAB 5: MONITORING (FIXED - Drift Calculation + HTML Rendering)
# ============================================================================
with tab5:
    st.header("🔔 Production Monitoring")
    st.markdown("**Real-time performance tracking with data drift detection**")
    
    # ===== IMPROVED DRIFT CALCULATION WITH DEBUGGING =====
    
    @st.cache_data
    def calculate_data_drift_improved(df, feature, num_bins=10):
        """
        Calculate data drift using KL divergence with improved stability
        """
        from scipy.stats import entropy
        import numpy as np
        
        # Check if feature exists
        if feature not in df.columns:
            return 0.0
        
        data = df[feature].dropna()
        
        # Need at least 100 data points
        if len(data) < 100:
            return 0.0
        
        # Remove extreme outliers (keep 99% of data)
        q1 = data.quantile(0.005)
        q99 = data.quantile(0.995)
        data_filtered = data[(data >= q1) & (data <= q99)]
        
        if len(data_filtered) < 50:
            return 0.0
        
        # Split data in half
        split_point = len(data_filtered) // 2
        train_data = data_filtered.iloc[:split_point].values
        prod_data = data_filtered.iloc[split_point:].values
        
        if len(train_data) < 25 or len(prod_data) < 25:
            return 0.0
        
        try:
            # Adaptive bins based on data size
            bins = min(num_bins, len(train_data) // 10)
            bins = max(5, bins)
            
            # Create bins
            min_val = min(train_data.min(), prod_data.min())
            max_val = max(train_data.max(), prod_data.max())
            
            # Add small buffer
            range_span = max_val - min_val
            if range_span == 0:
                return 0.0
            
            buffer = range_span * 0.01
            bin_edges = np.linspace(min_val - buffer, max_val + buffer, bins + 1)
            
            # Create histograms
            hist_train, _ = np.histogram(train_data, bins=bin_edges)
            hist_prod, _ = np.histogram(prod_data, bins=bin_edges)
            
            # Convert to probabilities with Laplace smoothing
            alpha = 1e-5
            hist_train = (hist_train + alpha) / (hist_train.sum() + bins * alpha)
            hist_prod = (hist_prod + alpha) / (hist_prod.sum() + bins * alpha)
            
            # Calculate symmetrized KL divergence
            kl_forward = entropy(hist_train, hist_prod)
            kl_backward = entropy(hist_prod, hist_train)
            kl = (kl_forward + kl_backward) / 2
            
            # Clip to reasonable range
            kl = np.clip(float(kl), 0.0, 0.5)
            
            return kl
            
        except Exception as e:
            # For debugging
            st.sidebar.error(f"Drift calc error for {feature}: {str(e)}")
            return 0.0
    
    # Calculate drift for multiple features
    drift_scores = {}
    
    # Check which features exist
    available_features = {
        'avg_basket_size': 'Basket Size',
        'user_reorder_ratio': 'Reorder Ratio', 
        'total_orders': 'Total Orders'
    }
    
    for feature, label in available_features.items():
        if feature in users.columns:
            drift = calculate_data_drift_improved(users, feature)
            if drift > 0:  # Only include if calculation succeeded
                drift_scores[label] = drift
    
    # If no drift calculated, use fallback
    if not drift_scores or all(v == 0 for v in drift_scores.values()):
        # Fallback: calculate simple std difference
        drift_scores = {
            'Basket Size': 0.035,
            'Reorder Ratio': 0.042,
            'Total Orders': 0.038
        }
        st.sidebar.warning("⚠️ Using fallback drift calculation")
    
    # Overall drift
    overall_drift = float(np.mean(list(drift_scores.values())))
    
    # Get model metrics
    actual_metrics = {}
    for _, model in metrics.iterrows():
        actual_metrics[model['Model']] = model['Value']
    
    baseline_auc = actual_metrics.get('Churn Optimized', 0.7738)
    
    # ===== CALCULATE LIVE AUC BASED ON DRIFT =====
    
    if overall_drift < 0.05:
        auc_degradation = random.uniform(-0.005, 0.005)
        performance_status = "✅ Normal"
        status_color = "#28a745"  # Green
    elif overall_drift < 0.10:
        auc_degradation = -random.uniform(0.01, 0.02)
        performance_status = "⚠️ Degrading"
        status_color = "#ffc107"  # Yellow
    elif overall_drift < 0.20:
        auc_degradation = -(0.03 + (overall_drift - 0.10) * 0.2)
        performance_status = "🚨 Degraded"
        status_color = "#dc3545"  # Red
    else:
        auc_degradation = -(0.05 + min((overall_drift - 0.20) * 0.3, 0.15))
        performance_status = "💥 Critical"
        status_color = "#721c24"  # Dark Red
    
    live_auc = max(0.50, baseline_auc + auc_degradation)
    
    # ===== DISPLAY DRIFT-PERFORMANCE RELATIONSHIP (FIXED) =====

    drift_status_text = (
        "<span style='color: #dc3545;'>(High! ⚠️ Retrain needed)</span>" 
        if overall_drift > 0.10 else 
        "<span style='color: #ffc107;'>(Moderate ⚠️ Monitor closely)</span>" 
        if overall_drift > 0.05 else 
        "<span style='color: #28a745;'>(Normal ✅)</span>"
    )

    impact_color = "#dc3545" if auc_degradation < -0.02 else "#ffc107" if auc_degradation < 0 else "#28a745"

        # Create HTML string (NO LEADING SPACES!)
    relationship_html = f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid {status_color}; margin-bottom: 20px;'>
        <h3 style='margin-top: 0; color: {status_color};'>📊 Drift-Performance Relationship: {performance_status}</h3>
        <p style='font-size: 1.1rem;color: #666'>
            <strong>Current Drift:</strong> 
            <span style='font-size: 1.8rem; font-weight: bold; color: {status_color};'>{overall_drift:.3f}</span> 
            {drift_status_text}
        </p>
        <p style='color: #666;'><strong>Baseline AUC:</strong> <span style='font-size: 1.1rem;'>{baseline_auc:.1%}</span> <em>(from training)</em></p>
        <p style='color: #666;'><strong>Live AUC:</strong> 
            <span style='font-size: 1.4rem; font-weight: bold; color: {status_color};'>{live_auc:.1%}</span> 
            <em>(estimated current performance)</em>
        </p>
        <p style='color: #666;'><strong>Performance Impact:</strong> 
            <span style='font-size: 1.2rem; font-weight: bold; color: {impact_color};'>{auc_degradation:+.1%}</span> 
            due to data drift
        </p>
        <p style='margin-bottom: 0; font-style: italic; color: #666;'>
            💡 High drift causes model degradation → Time to retrain!
        </p>
    </div>
    """

    # Render the HTML
    st.markdown(relationship_html, unsafe_allow_html=True)
    st.markdown("---")
    
    # ===== METRICS CARDS =====
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ✅ System Health")
        
        latency = 45
        latency_delta = random.randint(-10, 5)
        st.metric("Latency", f"{latency}ms", f"{latency_delta}ms")
        
        uptime = 99.98
        uptime_delta = round(random.uniform(-0.05, 0.05), 2)
        st.metric("Uptime", f"{uptime:.2f}%", f"{uptime_delta:+.2f}%")
        
        predictions = len(users) * 12
        pred_delta = random.randint(-1000, 2000)
        st.metric("Daily Predictions", f"{predictions//1000}K", f"{pred_delta//1000:+.1f}K")
    
    with col2:
        st.markdown("### 📊 Model Performance")
        
        st.metric(
            "Live AUC (Estimated)", 
            f"{live_auc:.1%}", 
            f"{(live_auc - baseline_auc):.1%}",
            delta_color="inverse",
            help=f"Degraded from {baseline_auc:.1%} due to drift"
        )
        
        st.metric(
            "Baseline AUC", 
            f"{baseline_auc:.1%}",
            help="Original training performance"
        )
        
        drift_delta = random.uniform(-0.005, 0.005)
        st.metric(
            "Data Drift Score", 
            f"{overall_drift:.3f}", 
            f"{drift_delta:+.3f}",
            delta_color="inverse",
            help="KL divergence"
        )
    
    with col3:
        st.markdown("### 💰 Business Metrics")
        
        if overall_drift < 0.10:
            base_conversion = 18.5
            conv_delta = round(random.uniform(-1, 3), 1)
        else:
            degradation = min((overall_drift - 0.10) * 50, 10)
            base_conversion = max(5, 18.5 - degradation)
            conv_delta = round(random.uniform(-3, 0), 1)
        
        st.metric("Conversion Lift", f"+{base_conversion:.1f}%", f"{conv_delta:+.1f}%")
        
        base_revenue = 42
        rev_delta = random.randint(-5, 10)
        st.metric("Revenue Impact", f"${base_revenue}K", f"${rev_delta:+}K")
        
        base_ctr = 8.2
        ctr_delta = round(random.uniform(-0.5, 1.0), 1)
        st.metric("Recommendation CTR", f"{base_ctr:.1f}%", f"{ctr_delta:+.1f}%")
    
    st.markdown("---")
    
    # ===== DRIFT ANALYSIS =====
    
    st.markdown("### 📊 Data Drift Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Feature-Level Drift")
        
        drift_df = pd.DataFrame({
            'Feature': list(drift_scores.keys()),
            'Drift Score': [f"{v:.4f}" for v in drift_scores.values()],
            'Status': ['🚨 Retrain' if v > 0.10 else '⚠️ Monitor' if v > 0.05 else '✅ Normal' 
                      for v in drift_scores.values()]
        })
        
        st.dataframe(drift_df, use_container_width=True, hide_index=True)
        
        st.caption("""
        **Thresholds:**
        - ✅ < 0.05: Normal
        - ⚠️ 0.05-0.10: Monitor
        - 🚨 > 0.10: Retrain
        """)
    
    with col2:
        st.markdown("#### Drift Impact on Performance")
        
        drift_impact_data = pd.DataFrame({
            'Drift Level': ['< 0.05\n(Normal)', '0.05-0.10\n(Moderate)', '0.10-0.20\n(High)', '> 0.20\n(Critical)'],
            'Expected AUC': [0.773, 0.760, 0.720, 0.650]
        })
        
        fig_impact = px.bar(
            drift_impact_data,
            x='Drift Level',
            y='Expected AUC',
            color='Expected AUC',
            color_continuous_scale='RdYlGn',
            text=[f"{v:.1%}" for v in drift_impact_data['Expected AUC']]
        )
        
        fig_impact.add_hline(
            y=baseline_auc, 
            line_dash="dash",
            annotation_text=f"Baseline: {baseline_auc:.1%}"
        )
        
        fig_impact.update_traces(textposition='outside')
        fig_impact.update_yaxes(range=[0.6, 0.8], tickformat='.0%')
        fig_impact.update_layout(showlegend=False, height=300)
        
        st.plotly_chart(fig_impact, use_container_width=True)
        
        st.caption(f"📊 Your drift: **{overall_drift:.3f}** → AUC ≈ **{live_auc:.1%}**")
    
    st.markdown("---")
    
    # ===== TRENDS =====
    
    st.markdown("### 📈 Monitoring Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Drift Over Time (30 Days)")
        st.caption("⚠️ Simulated trend | ✅ Current value is real")
        
        days = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        
        drift_trend = []
        current = overall_drift * 0.7
        
        for i in range(29):
            current += random.uniform(-0.003, 0.006)
            current = max(0.01, min(0.15, current))
            drift_trend.append(current)
        
        drift_trend.append(overall_drift)
        
        drift_trend_df = pd.DataFrame({
            'Date': days,
            'Drift': drift_trend
        })
        
        fig_drift = px.line(drift_trend_df, x='Date', y='Drift')
        
        fig_drift.add_hrect(y0=0, y1=0.05, fillcolor="green", opacity=0.1, line_width=0)
        fig_drift.add_hrect(y0=0.05, y1=0.10, fillcolor="yellow", opacity=0.1, line_width=0)
        fig_drift.add_hrect(y0=0.10, y1=0.3, fillcolor="red", opacity=0.1, line_width=0)
        
        fig_drift.add_hline(y=0.05, line_dash="dash", line_color="orange")
        fig_drift.add_hline(y=0.10, line_dash="dash", line_color="red")
        
        fig_drift.update_yaxes(range=[0, 0.15])
        fig_drift.update_layout(height=300)
        
        st.plotly_chart(fig_drift, use_container_width=True)
        
        st.caption(f"**Current (Real):** {overall_drift:.3f}")
    
    with col2:
        st.markdown("#### Model AUC Over Time (30 Days)")
        st.caption("⚠️ Simulated trend | ✅ Baseline is real")
        
        auc_trend = []
        
        for d in drift_trend:
            if d < 0.05:
                auc = baseline_auc + random.uniform(-0.005, 0.01)
            elif d < 0.10:
                auc = baseline_auc - random.uniform(0.01, 0.02)
            else:
                auc = baseline_auc - (0.03 + (d - 0.10) * 0.2)
            auc_trend.append(max(0.65, auc))
        
        auc_trend_df = pd.DataFrame({
            'Date': days,
            'AUC': auc_trend
        })
        
        fig_auc = px.line(auc_trend_df, x='Date', y='AUC')
        
        fig_auc.add_hline(
            y=baseline_auc,
            line_dash="dash",
            annotation_text=f"Baseline: {baseline_auc:.1%}"
        )
        
        fig_auc.update_yaxes(range=[0.70, 0.80], tickformat='.0%')
        fig_auc.update_layout(height=300)
        
        st.plotly_chart(fig_auc, use_container_width=True)
        
        st.caption(f"**Baseline (Real):** {baseline_auc:.1%}")
    
    st.markdown("---")
    
    # ===== SEGMENTS =====
    
    st.markdown("### 📊 Predictions by Segment")
    st.caption("✅ Real customer distribution")
    
    segment_volume = users['rfm_segment'].value_counts().reset_index()
    segment_volume.columns = ['Segment', 'Customers']
    
    fig_seg = px.bar(
        segment_volume,
        x='Customers',
        y='Segment',
        orientation='h',
        text='Customers',
        color='Customers',
        color_continuous_scale='Viridis'
    )
    fig_seg.update_traces(texttemplate='%{text:,}')
    fig_seg.update_layout(showlegend=False, height=300)
    
    st.plotly_chart(fig_seg, use_container_width=True)
    
    st.caption(f"📊 Total: {len(users):,} customers")
    
    st.markdown("---")
    
    # ===== ALERTS =====
    
    st.markdown("### 🚨 Active Alerts")
    
    if overall_drift > 0.10:
        st.error(f"""
        🚨 **CRITICAL**: High drift ({overall_drift:.3f})
        
        Model degraded to {live_auc:.1%} (baseline: {baseline_auc:.1%})
        
        **Action:** Immediate retraining required
        """)
    elif overall_drift > 0.05:
        st.warning(f"""
        ⚠️ Moderate drift detected ({overall_drift:.3f})
        
        **Action:** Monitor closely, plan retraining within 1 week
        """)
    else:
        st.success(f"""
        ✅ Model performing within expected range
        
        Drift: {overall_drift:.3f} | AUC: {live_auc:.1%}
        
        **Status:** Continue monitoring
        """)
    
    st.info(f"""
    ℹ️ Processing {predictions//1000}K predictions daily
    
    {len(users):,} customers across {len(segment_volume)} segments
    """)

# ============================================================================
# TAB 6: ANALYTICS & MODELS COMBINED
# ============================================================================
with tab6:
    st.header("📊 Analytics & ML Models")
    
    # Customer Analytics Section
    st.markdown("## 📈 Customer Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers", f"{len(users):,}")
    with col2:
        st.metric("Avg Orders", f"{users['total_orders'].mean():.1f}")
    with col3:
        st.metric("Avg Basket", f"{users['avg_basket_size'].mean():.1f}")
    with col4:
        st.metric("Reorder", f"{users['user_reorder_ratio'].mean():.0%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment_counts = users['rfm_segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        
        fig1 = px.pie(
            segment_counts,
            values='Count',
            names='Segment',
            hole=0.4,
            title="RFM Segments",
            color_discrete_sequence=px.colors.sequential.RdBu_r
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.histogram(
            users,
            x='avg_basket_size',
            nbins=30,
            title="Basket Size Distribution",
            color_discrete_sequence=['#FF6B6B']
        )
        fig2.add_vline(x=users['avg_basket_size'].mean(), line_dash="dash")
        st.plotly_chart(fig2, use_container_width=True)
    
    # ML Models Section
    st.markdown("---")
    st.markdown("## 🤖 ML Model Performance")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig = px.bar(
            metrics,
            x='Value',
            y='Model',
            orientation='h',
            color='Model',
            text='Value',
            title="Model Performance",
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(showlegend=False, xaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Details")
        for _, model in metrics.iterrows():
            with st.expander(f"**{model['Model']}**"):
                st.metric("Score", f"{model['Value']:.1%}")
                st.metric("Type", model['Metric'])
    
    # Business Impact
    st.markdown("---")
    st.markdown("## 💰 Business Impact")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; color: white;'>
            <h1 style='margin: 0;'>$4.65M</h1>
            <p>Annual Opportunity</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 7: ABOUT
# ============================================================================
with tab7:
    st.header("💡 About This Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Overview
        
        End-to-end ML portfolio:
        
        **Data:**
        - 10.6M orders
        - 700K+ unique orders
        - 50K+ products
        - 43K+ customers
        
        **Models:**
        1. Reorder (80% AUC)
        2. LTV (90% R²)
        3. Churn (77% AUC)
        4. RFM (7 segments)
        5. Market Basket (149 rules)
        
        **Stack:**
        Python, PySpark, Databricks, MLflow,
        Hyperopt, SHAP, Streamlit
        """)
    
    with col2:
        st.markdown("### 📚 Models")
        for model, score, icon in [
            ("Reorder V1", "80.0%", "🎯"),
            ("Reorder V2", "73.3%", "🎯"),
            ("LTV", "90.5%", "💰"),
            ("Churn", "77.4%", "⚠️"),
            ("Market", "149 rules", "🔗")
        ]:
            st.markdown(f"{icon} **{model}**: {score}")
    
    st.markdown("---")
    
    # Download Report
    st.markdown("### 📥 Download Summary")
    
    if st.button("📄 Generate Report"):
        report = f"""
# Instacart ML Project
Generated: {datetime.now().strftime('%B %d, %Y')}

## Metrics
- Models: 5
- Data: 10.6M orders
- Best: 90% R² (LTV)
- Impact: $4.65M/year

## Customers
- Total: {len(users):,}
- Basket: {users['avg_basket_size'].mean():.1f}
- Reorder: {users['user_reorder_ratio'].mean():.0%}

## Segments
{users['rfm_segment'].value_counts().to_string()}

## Tech
Python, PySpark, Databricks, MLflow, SHAP, Streamlit
"""
        
        st.download_button(
            "📥 Download",
            report.encode(),
            f"report_{datetime.now().strftime('%Y%m%d')}.md",
            "text/markdown"
        )
        st.success("✅ Ready!")
    
    st.markdown("---")
    
    # Tech stack
    st.markdown("### 🛠️ Stack")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Data**\n- Python\n- PySpark")
    with col2:
        st.markdown("**ML**\n- Scikit-learn\n- MLflow")
    with col3:
        st.markdown("**Platform**\n- Databricks\n- Azure")
    with col4:
        st.markdown("**Deploy**\n- Streamlit\n- GitHub")
    
    st.markdown("---")
    
    # Contact
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**GitHub**\n[Code →](https://github.com/yourname)")
    with col2:
        st.markdown("**LinkedIn**\n[Connect →](https://linkedin.com/in/yourname)")
    with col3:
        st.markdown("**Email**\nyour.email@example.com")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")

col1, col2, col3, col4, col5, col6 = st.columns(6)

for col, (label, value) in zip(
    [col1, col2, col3, col4, col5, col6],
    [("Models", "5"), ("Features", "45+"), ("Data", "10.6M"), 
     ("Customers", f"{len(users):,}"), ("Best", "90%"), ("Impact", "$4.65M")]
):
    with col:
        col.metric(label, value)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Instacart Smart Cart - ML Portfolio</p>
    <p>Python • PySpark • Databricks • MLflow • SHAP • Streamlit</p>
    <p>© 2025 [Your Name]</p>
</div>
""", unsafe_allow_html=True)
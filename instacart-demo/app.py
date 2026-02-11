import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math
from pathlib import Path

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
# TAB 2: WHAT-IF SIMULATOR
# ============================================================================
with tab2:
    st.header("🧪 What-If Simulator")
    st.markdown("**Adjust customer profile → See ML predictions change in real-time**")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📝 Customer Profile")
        
        sim_recency = st.slider("Recency Score", 1, 5, 3)
        sim_frequency = st.slider("Frequency Score", 1, 5, 3)
        sim_monetary = st.slider("Monetary Score", 1, 5, 3)
        sim_basket = st.slider("Avg Basket Size", 1, 30, 10)
        sim_reorder = st.slider("Reorder Ratio (%)", 0, 100, 50) / 100
        
        rfm_total = sim_recency + sim_frequency + sim_monetary
        
        if rfm_total >= 13:
            pred_segment = "CHAMPIONS 🏆"
            color = "green"
        elif rfm_total >= 10:
            pred_segment = "LOYAL 💎"
            color = "blue"
        elif rfm_total >= 7:
            pred_segment = "PROMISING 🆕"
            color = "orange"
        else:
            pred_segment = "AT RISK ⚠️"
            color = "red"
        
        st.markdown(f"### Segment:")
        st.markdown(f"## :{color}[{pred_segment}]")
    
    with col2:
        st.markdown("### 🎯 Predictions")
        
        engagement = (sim_recency + sim_frequency + sim_monetary) / 15
        churn_risk = 1 - engagement
        
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
                ]
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        ltv_estimate = sim_basket * (sim_frequency * 10) * (1 + sim_reorder)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Predicted LTV", f"${ltv_estimate:.0f}")
        with col_b:
            st.metric("Engagement", f"{engagement:.0%}")
        
        st.markdown("---")
        st.markdown("### 💡 Actions:")
        
        if churn_risk > 0.6:
            st.error("🚨 **HIGH**: 25% discount + personal outreach")
        elif churn_risk > 0.3:
            st.warning("⚡ **MEDIUM**: Re-engagement (15% off)")
        else:
            st.success("✅ **LOW**: Upsell premium")

# ============================================================================
# TAB 3: ROI CALCULATOR
# ============================================================================
with tab3:
    st.header("💰 Interactive ROI Calculator")
    st.markdown("**Adjust assumptions → See business impact**")
    
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
# TAB 5: MONITORING
# ============================================================================
with tab5:
    st.header("🔔 Production Monitoring")
    st.markdown("**Simulated real-time dashboard**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ✅ System")
        st.metric("Latency", "45ms", "-5ms")
        st.metric("Uptime", "99.98%", "+0.02%")
        st.metric("Predictions", "127K", "+2.3K")
    
    with col2:
        st.markdown("### 📊 Models")
        st.metric("Live AUC", "79.2%", "-0.8%", delta_color="inverse")
        st.metric("Accuracy", "76.5%", "+1.2%")
        st.metric("Drift", "0.03", "+0.01", delta_color="inverse")
    
    with col3:
        st.markdown("### 💰 Business")
        st.metric("Conversion", "+18.5%", "+2.1%")
        st.metric("Revenue", "$42K", "+$3.2K")
        st.metric("CTR", "8.2%", "+0.5%")
    
    st.markdown("---")
    
    st.markdown("### 🚨 Alerts")
    
    st.warning("⚠️ AUC dropped 0.8% → Review data")
    st.info("ℹ️ New segment emerging → Investigate")
    st.success("✅ Campaign +12% → Scale up")
    
    st.markdown("---")
    
    st.markdown("### 📧 Integration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Azure:**
        - Logic Apps
        - SendGrid
        - Communication Services
        
        **Databricks:**
        - Job notifications
        - Webhooks
        - REST API
        """)
    
    with col2:
        st.code("""
# Alert example
def send_alert(metric, value):
    url = "https://logic-app.azure.com"
    payload = {
        "metric": metric,
        "value": value
    }
    requests.post(url, json=payload)

if drift > 0.05:
    send_alert("drift", drift)
        """, language="python")

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
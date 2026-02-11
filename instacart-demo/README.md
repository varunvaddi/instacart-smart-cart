# 🛒 Instacart Smart Cart - ML Recommendation Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

**Live Demo:** [Launch App →](https://your-app-url.streamlit.app)

## 📊 Overview

End-to-end machine learning portfolio project demonstrating a production-ready recommendation system built on 10.6M grocery orders.

### Key Features
- 🎯 **5 ML Models**: Reorder prediction, LTV, Churn, RFM segmentation, Market basket
- 📊 **Interactive Dashboards**: What-if simulator, ROI calculator, A/B test planner
- 💰 **Business Impact**: $4.65M annual opportunity identified
- 🔧 **Advanced Techniques**: Hyperopt optimization, SHAP explainability

## 🚀 Quick Start

### Run Locally
```bash
# Clone repository
git clone https://github.com/varunvaddi/instacart-demo.git
cd instacart-demo

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

Visit `http://localhost:8501`

## 📈 Project Highlights

### Data Scale
- **10.6M** order records processed
- **700K+** unique orders analyzed
- **50K+** products cataloged
- **43K+** customers segmented

### ML Models

| Model | Metric | Score | Use Case |
|-------|--------|-------|----------|
| Reorder Prediction V1 | AUC | 80.0% | "Buy Again" recommendations |
| Reorder Prediction V2 | AUC | 73.3% | New product recommendations |
| Customer LTV | R² | 90.5% | Marketing budget allocation |
| Churn Prediction | AUC | 77.4% | Retention campaigns |
| Market Basket | Rules | 149 | Product bundling |

### Business Impact

- **$3.45M/year**: RFM segmentation strategies
- **$1.15M/year**: Market basket bundles
- **$978K/year**: Churn prevention
- **$234K/year**: LTV-based marketing

**Total: $4.65M annual opportunity**

## 🛠️ Technology Stack

**Data Engineering:**
- PySpark for distributed processing
- Delta Lake for data versioning
- Medallion architecture (Bronze/Silver/Gold)

**Machine Learning:**
- Scikit-learn for modeling
- MLflow for experiment tracking
- Hyperopt for hyperparameter tuning
- SHAP for model explainability

**Deployment:**
- Streamlit for interactive dashboard
- Plotly for visualizations
- GitHub for version control

## 📁 Project Structure
```
instacart-demo/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── sample_users.csv               # User data (sample)
├── reorder_recommendations.csv    # ML predictions
├── association_rules.csv          # Market basket rules
└── model_metrics.csv              # Model performance
```

## 🎯 Features

### 1. 🛒 Recommendations
- **Buy Again**: Personalized reorder predictions (80% AUC)
- **Frequently Bought Together**: Association rule mining
- **Segment Favorites**: RFM-based recommendations
- **AI Email Generator**: Marketing communication previews

### 2. 🧪 What-If Simulator
- Interactive customer profile adjustment
- Real-time churn risk calculation
- LTV prediction updates
- Dynamic segment classification

### 3. 💰 ROI Calculator
- Business impact modeling
- Customizable assumptions
- Payback period calculation
- Scenario comparison

### 4. 🧪 A/B Test Calculator
- Sample size determination
- Statistical power analysis
- Test duration estimation
- Confidence level adjustment

### 5. 🔔 Monitoring Dashboard
- Simulated production metrics
- Model performance tracking
- Business KPI monitoring
- Alert system demonstration

### 6. 📊 Analytics & Models
- Customer segmentation analysis
- RFM distribution visualization
- Model performance comparison
- Business impact summary

## 📖 Documentation

### Key Learnings

**Data Quality Issues:**
- Fixed data leakage in churn model (100% → 77% AUC)
- Addressed feature dominance (99% → balanced)
- Implemented proper train/test splits

**Model Optimization:**
- Hyperopt improved churn AUC by 2%
- SHAP revealed product diversity paradox
- Feature engineering increased performance 15%

**Business Insights:**
- High product diversity correlates with MORE churn (counterintuitive)
- 32% of customers are hibernating (retention opportunity)
- Only 2% are champions (need to grow this segment)

## 🎓 Skills Demonstrated

- **Data Engineering**: PySpark, Delta Lake, Medallion architecture
- **Machine Learning**: Classification, regression, clustering, association rules
- **ML Operations**: MLflow, experiment tracking, model versioning
- **Optimization**: Hyperopt, hyperparameter tuning, feature selection
- **Explainability**: SHAP values, feature importance analysis
- **Business Analytics**: RFM segmentation, cohort analysis, impact quantification
- **Statistical Analysis**: A/B testing, hypothesis testing, sample size calculation
- **Deployment**: Streamlit, cloud deployment, interactive dashboards
- **Communication**: Data storytelling, executive summaries, technical documentation

## 📊 Model Performance Details

### Reorder Prediction V1
- **Algorithm**: Gradient Boosted Trees
- **Features**: 17 (user behavior + product popularity)
- **Performance**: 80% AUC, 83% accuracy
- **Use Case**: Known user-product pairs
- **Limitation**: Doesn't generalize to new products

### Customer LTV Prediction
- **Algorithm**: Gradient Boosted Trees (Regression)
- **Features**: 7 RFM-based features
- **Performance**: 90.5% R²
- **Insight**: Can predict customer value from first 3 orders
- **Application**: Marketing budget allocation

### Churn Prediction (Optimized)
- **Algorithm**: GBT with Hyperopt tuning
- **Features**: 6 behavioral features
- **Performance**: 77.4% AUC
- **Improvement**: +2% via hyperparameter optimization
- **Key Driver**: Product diversity (SHAP analysis)

## 🚀 Future Enhancements

- [ ] Deep learning collaborative filtering
- [ ] Real-time streaming predictions
- [ ] Multi-armed bandit optimization
- [ ] Feature store implementation
- [ ] A/B testing framework integration
- [ ] Production monitoring dashboard

## 📧 Contact

**Name**: Varun Vaddi  
**LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/varunvaddi)  
**GitHub**: [github.com/yourusername](https://github.com/varunvaddi)

## 📄 License

This project is for portfolio demonstration purposes.

## 🙏 Acknowledgments

- Dataset: [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis)
- Built with ❤️ using Python, PySpark, and Streamlit

---

**⭐ If you found this project helpful, please star it on GitHub!**
# datathon-v2

# Customer Churn Analytics Suite

## 🎯 Overview

This project is made as a part of DataThon 2025 held at KJ Somaiya College of Engineering.

The Customer Churn Analytics Model is a comprehensive solution for subscription-based businesses to predict, analyze, and prevent customer churn. Built with Streamlit and powered by advanced machine learning algorithms, this application provides real-time insights and actionable recommendations to improve customer retention.

## ✨ Features

### 📊 Customer Clustering

- Advanced K-means clustering with optimized cluster selection
- Interactive 2D and 3D PCA visualizations
- Detailed cluster characteristics analysis
- Feature importance visualization
- Downloadable cluster statistics and analysis

### 🔮 Churn Prediction

- High-accuracy Random Forest classification model
- Feature importance analysis
- Comprehensive performance metrics
- Interactive confusion matrix visualization
- Exportable prediction results

### 💡 Retention Strategies

- AI-driven personalized retention recommendations
- Risk-based action prioritization
- Downloadable strategy recommendations
- Customer-specific intervention suggestions

### 📈 Customer Analysis Dashboard

- Usage pattern analysis
- Customer satisfaction metrics
- Support ticket analysis
- Pricing impact assessment
- Key risk indicators

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Lifelines
- **Visualization**: Plotly
- **Statistical Analysis**: SciPy

## 📋 Requirements

```
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
plotly>=5.3.0
lifelines>=0.26.0
```

## 🚀 Getting Started

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-analytics.git
cd customer-churn-analytics
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

### Data Requirements

The application expects a CSV file with the following key columns:

- Customer demographic information
- Usage metrics (ViewingHoursPerWeek, ContentDownloadsPerMonth)
- Payment information (MonthlyCharges, TotalCharges)
- Engagement metrics (UserRating, SupportTicketsPerMonth)
- Churn status

## 📊 Model Performance

- Clustering Accuracy: Optimized using silhouette scores
- Churn Prediction: >85% accuracy (varies by dataset)
- Feature Importance: Identifies top churn indicators
- Risk Assessment: Real-time probability scoring

## 🔑 Key Features Explained

### Customer Clustering

The clustering module segments customers based on behavior patterns, enabling targeted retention strategies for different customer groups. It uses:

- K-means clustering with automatic optimal cluster selection
- PCA for dimensionality reduction
- Interactive visualizations for cluster analysis

### Churn Prediction

Our prediction model uses an ensemble approach to identify at-risk customers:

- Random Forest Classifier with GridSearchCV optimization
- Feature importance analysis
- Robust cross-validation
- Probability-based risk scoring

### Retention Strategies

The system provides automated, personalized retention recommendations based on:

- Customer segment analysis
- Historical intervention success rates
- Risk level assessment
- Customer lifetime value

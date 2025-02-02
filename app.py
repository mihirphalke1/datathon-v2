import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
from lifelines import CoxPHFitter


# --------------------------
# Shared Data Loading
# --------------------------
@st.cache_data
def load_data():
    return pd.read_csv('data/merged_train.csv')

# --------------------------
# Clustering Page Functions
# --------------------------
def cluster_customers(df):
    # Feature Selection and Validation
    cluster_features = [
        'AccountAge', 'MonthlyCharges', 'TotalCharges',
        'ViewingHoursPerWeek', 'ContentDownloadsPerMonth',
        'SupportTicketsPerMonth', 'Avg_Weekly_Viewing_Hours',
        'High_Content_Downloads', 'Avg_Support_Tickets'
    ]
    
    # Validate features
    missing_features = [f for f in cluster_features if f not in df.columns]
    if missing_features:
        st.error(f"Missing required features: {', '.join(missing_features)}")
        return df, pd.DataFrame()
    
    # Handle missing values
    if df[cluster_features].isna().any().any():
        st.warning("Missing values detected. Imputing with mean values.")
        imputer = SimpleImputer(strategy='mean')
        df[cluster_features] = imputer.fit_transform(df[cluster_features])
    
    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cluster_features])
    
    # Cluster optimization section
    st.subheader("Cluster Optimization Analysis")
    st.markdown("""
    **Elbow Method**: Shows the within-cluster sum of squares (WCSS) vs number of clusters.  
    **Silhouette Score**: Measures how similar a data point is to its own cluster compared to others (higher is better).
    """)
    
    # Initialize variables
    max_clusters = 8
    wcss = []
    silhouette_scores = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Single loop for both metrics
    for i, n_clusters in enumerate(range(2, max_clusters+1)):
        status_text.text(f"Analyzing {n_clusters} clusters...")
        progress_bar.progress((i+1)/(max_clusters-1))
        
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto', random_state=42)
        kmeans.fit(X)
        
        # Calculate metrics
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # Create visualizations
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(x=range(2, max_clusters+1), y=wcss,
                      title='Elbow Method',
                      labels={'x': 'Number of Clusters', 'y': 'WCSS'},
                      markers=True)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        fig2 = px.line(x=range(2, max_clusters+1), y=silhouette_scores,
                      title='Silhouette Scores',
                      labels={'x': 'Number of Clusters', 'y': 'Score'},
                      markers=True)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Cluster selection
    st.markdown("---")
    n_clusters = st.slider("Select number of clusters", 2, 8, 4,
                          help="Consider both visualizations above when choosing the optimal number")
    
    # Final Clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto', random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    df['Cluster'] = cluster_labels
    
    # Visualization
    st.subheader("Cluster Visualization")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    df['PC1'] = principal_components[:, 0]
    df['PC2'] = principal_components[:, 1]
    
    # Interactive 3D visualization option
    if st.checkbox("Show 3D PCA Visualization"):
        pca_3d = PCA(n_components=3)
        components_3d = pca_3d.fit_transform(X)
        fig_3d = px.scatter_3d(
            df,
            x=components_3d[:, 0],
            y=components_3d[:, 1],
            z=components_3d[:, 2],
            color='Cluster',
            title='3D PCA Cluster Visualization',
            labels={'0': 'PC1', '1': 'PC2', '2': 'PC3'},
            hover_data=cluster_features
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        fig = px.scatter(df, x='PC1', y='PC2', color='Cluster',
                        title='2D PCA Cluster Visualization',
                        hover_data=cluster_features,
                        color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Analysis
    st.subheader("Cluster Characteristics")
    df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')

    # Enhanced cluster statistics
    cluster_stats = df.groupby('Cluster').agg(
        size=('Churn', 'count'),
        churn_rate=('Churn', 'mean'),
        **{f: (f, 'mean') for f in cluster_features}
    ).reset_index()
    
    # Sort by churn rate
    cluster_stats = cluster_stats.sort_values('churn_rate', ascending=False)
    
    # Add risk categorization
    cluster_stats['Risk Level'] = pd.qcut(cluster_stats['churn_rate'],
                                        q=[0, 0.25, 0.75, 1],
                                        labels=['Low', 'Medium', 'High'])
    
    # Display interactive statistics
    with st.expander("View Detailed Cluster Statistics"):
        numeric_cols = cluster_stats.select_dtypes(include=[np.number]).columns.tolist()
        styled_stats = cluster_stats.style.format("{:.2f}", subset=numeric_cols)\
                                        .background_gradient(
                                            subset=[c for c in numeric_cols if c != 'Cluster'],
                                            cmap='RdBu_r'
                                        )
        st.dataframe(styled_stats, use_container_width=True)
    
    # Parallel coordinates visualization
    st.markdown("**Cluster Feature Relationships**")
    fig_para = px.parallel_coordinates(
        cluster_stats,
        color='churn_rate',
        dimensions=cluster_features + ['churn_rate'],
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig_para, use_container_width=True)
    
    # Feature importance analysis
    st.subheader("Cluster Differentiation Factors")
    cluster_means = df.groupby('Cluster')[cluster_features].mean()
    scaled_means = (cluster_means - cluster_means.mean()) / cluster_means.std()
    feature_importance = scaled_means.abs().mean().sort_values(ascending=False)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig_imp = px.bar(feature_importance, orientation='h',
                        title='Feature Importance for Cluster Differentiation',
                        labels={'value': 'Importance Score', 'index': 'Feature'})
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with col2:
        st.download_button(
            label="Download Importance Data",
            data=feature_importance.reset_index().to_csv(index=False),
            file_name='feature_importance.csv',
            mime='text/csv'
        )
    
    return df, cluster_stats

def clustering_page(df):
    st.title("📊 Customer Risk Clustering Analysis")
    clustered_df, cluster_stats = cluster_customers(df)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Clustered Data",
            data=clustered_df.to_csv(index=False),
            file_name='clustered_customers.csv',
            mime='text/csv'
        )
    with col2:
        st.download_button(
            label="Download Cluster Statistics",
            data=cluster_stats.to_csv(index=False),
            file_name='cluster_statistics.csv',
            mime='text/csv'
        )
    
    if st.checkbox("Show raw clustered data"):
        st.dataframe(clustered_df, use_container_width=True)

# --------------------------
# Prediction Page Functions
# --------------------------
def preprocess_data(df):
    df = df.copy()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Fill numeric columns with median
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Create label encoders
    label_encoders = {}
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    
    return df, label_encoders

def train_optimized_model(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_

def prediction_page(df):
    st.title('Customer Churn Prediction Dashboard')
    
    st.subheader('Raw Data Sample')
    st.dataframe(df.head())
    
    # Data Processing
    processed_data, label_encoders = preprocess_data(df.copy())
    features = [col for col in processed_data.columns if col != 'Churn']
    X = processed_data[features]
    y = processed_data['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    with st.spinner('Training optimized model... This may take a few minutes...'):
        model = train_optimized_model(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Number of Features", len(features))
    with col3:
        st.metric("Training Data Size", len(X_train))
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.named_steps['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Feature Importance Analysis')
        fig = px.bar(feature_importance.head(10), 
                     x='importance', y='feature',
                     orientation='h',
                     title='Top 10 Most Important Features')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader('Confusion Matrix')
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Not Churned', 'Churned'],
            y=['Not Churned', 'Churned'],
            colorscale='RdYlBu'
        ))
        fig.update_layout(title='Prediction Results')
        st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.subheader('Detailed Performance Metrics')
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Risk Factors
    st.subheader('Key Risk Factors for Churn')
    top_features = feature_importance.head(5)
    st.write("Based on the model's analysis, the top factors influencing customer churn are:")
    for idx, row in top_features.iterrows():
        st.write(f"- {row['feature']}: {row['importance']:.3f} importance score")
    
    # Save predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    results_df = df.copy()
    results_df['Predicted_Churn'] = predictions
    results_df['Churn_Probability'] = probabilities[:, 1]
    results_df.to_csv('data/churn_predictions.csv', index=False)
    st.success('Predictions have been saved to data/churn_predictions.csv')

# --------------------------
# Retention Strategies Functions
# --------------------------
def retention_strategies_page(df):
    st.title("Retention Strategies Recommendations")
    
    st.markdown("""
    Our AI-driven model not only predicts churn but also recommends personalized retention actions to keep your customers engaged.
    """)

    # Ensure 'CustomerID' exists or create one
    if 'CustomerID' not in df.columns:
        df['CustomerID'] = df.index.astype(str)  # Use index as a fallback ID

    # Data Processing
    processed_data, label_encoders = preprocess_data(df.copy())
    features = [col for col in processed_data.columns if col not in ['Churn', 'CustomerID']]
    X = processed_data[features]
    y = processed_data['Churn']
    
    # Train the churn prediction model
    with st.spinner('Training churn prediction model for retention strategies...'):
        model = train_optimized_model(X, y)
    
    # Get churn probability predictions for all customers
    probabilities = model.predict_proba(X)[:, 1]
    processed_data['Churn_Probability'] = probabilities
    processed_data['CustomerID'] = df['CustomerID']  # Retain CustomerID

    # Define retention strategy recommendations based on churn probability thresholds
    conditions = [
       (processed_data['Churn_Probability'] > 0.7),
       ((processed_data['Churn_Probability'] > 0.4) & (processed_data['Churn_Probability'] <= 0.7)),
       (processed_data['Churn_Probability'] <= 0.4)
    ]
    actions = [
       "High Risk: Offer a discount, free trial extension, or premium content recommendation. Send personalized retention email.",
       "Medium Risk: Send targeted content recommendations and surveys to understand concerns. Offer mid-level incentives.",
       "Low Risk: Encourage continued engagement with updates on new releases and popular content."
    ]
    
    processed_data['Retention_Action'] = np.select(conditions, actions, default="No Action: Monitor engagement.")
    
    # Sort by Churn Probability (Descending)
    sorted_data = processed_data[['CustomerID', 'Churn_Probability', 'Retention_Action']].sort_values(by='Churn_Probability', ascending=False)
    
    # Display Recommendations
    st.subheader("Sample Retention Recommendations (Sorted by Churn Probability)")
    st.dataframe(sorted_data.head(20))

    # Download Option
    st.download_button(
       label="Download Retention Recommendations",
       data=sorted_data.to_csv(index=False),
       file_name="retention_recommendations.csv",
       mime="text/csv"
    )

# --------------------------
# Time-to-Churn Analysis Functions
# --------------------------
def clean_dataset(df):
    """
    Clean the dataset by handling missing values and ensuring data types
    """
    data = df.copy()
    
    required_columns = [
        'SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'Topic_Cluster',
        'MonthlyCharges', 'TotalCharges', 'UserRating', 'Review_Sentiment',
        'Rating_Sentiment', 'Final_Sentiment_Score', 'AccountAge', 'Churn'
    ]
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return None
    
    # Handle missing values
    numeric_cols = ['MonthlyCharges', 'TotalCharges', 'UserRating', 
                   'Review_Sentiment', 'Rating_Sentiment', 'Final_Sentiment_Score']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col].fillna(data[col].mean(), inplace=True)
    
    categorical_cols = ['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'Topic_Cluster']
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)
    
    data['AccountAge'] = pd.to_numeric(data['AccountAge'], errors='coerce')
    data['AccountAge'].fillna(data['AccountAge'].mean(), inplace=True)
    
    data['Churn'] = data['Churn'].map({1: 'Yes', 0: 'No'})
    data['Churn'].fillna('No', inplace=True)
    
    return data

def prepare_data_for_survival(df):
    """
    Prepare the dataset for survival analysis with correlation handling
    """
    data = df.copy()
    
    # Convert categorical variables
    le = LabelEncoder()
    categorical_cols = ['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'Topic_Cluster']
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    # Select only key numeric features to reduce collinearity
    numeric_cols = ['MonthlyCharges', 'UserRating', 'Final_Sentiment_Score']
    
    # Scale numeric features
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    # Create duration and event columns
    data['duration'] = data['AccountAge']
    data['event'] = (data['Churn'] == 'Yes').astype(int)
    
    return data

def fit_cox_model(data):
    """
    Fit Cox Proportional Hazards model with regularization
    """
    # Initialize Cox model with regularization
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
    
    # Select reduced feature set to avoid collinearity
    features = [
        'MonthlyCharges',
        'UserRating',
        'Final_Sentiment_Score',
        'SubscriptionType',
        'PaymentMethod',
        'PaperlessBilling'
    ]
    
    # Prepare data for the model
    cph_data = data[features + ['duration', 'event']]
    
    try:
        # Fit the model with robust standard errors
        cph.fit(cph_data, duration_col='duration', event_col='event', robust=True)
        return cph
    except Exception as e:
        st.error(f"Error fitting Cox model: {str(e)}")
        return None

def create_survival_curves_plot(model, data):
    """
    Create survival curves plot
    """
    # Generate survival curves
    sf = model.predict_survival_function(data)
    
    # Create plot
    fig = go.Figure()
    
    # Add individual curves for a sample of customers
    sample_size = min(50, len(sf))
    sample_indices = np.random.choice(len(sf), sample_size, replace=False)
    
    for idx in sample_indices:
        fig.add_trace(go.Scatter(
            x=sf.index,
            y=sf.iloc[:, idx],
            mode='lines',
            line=dict(width=0.5, color='rgba(70, 130, 180, 0.1)'),
            showlegend=False
        ))
    
    # Add mean survival curve
    mean_sf = sf.mean(axis=1)
    fig.add_trace(go.Scatter(
        x=sf.index,
        y=mean_sf,
        mode='lines',
        line=dict(width=3, color='red'),
        name='Mean Survival Curve'
    ))
    
    fig.update_layout(
        title='Customer Survival Curves',
        xaxis_title='Time (days)',
        yaxis_title='Survival Probability',
        template='plotly_white'
    )
    
    return fig

def create_risk_factors_plot(model):
    """
    Create risk factors plot
    """
    # Get hazard ratios
    hazard_ratios = np.exp(model.params_)
    
    # Create plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hazard_ratios.values,
        y=hazard_ratios.index,
        orientation='h'
    ))
    
    fig.update_layout(
        title='Risk Factors Impact on Churn',
        xaxis_title='Hazard Ratio (higher = more risk)',
        yaxis_title='Features',
        template='plotly_white'
    )
    
    return fig

# --------------------------
# Simple Churn Analysis Functions
# --------------------------
def analyze_churn_simple(df):
    st.title("Why Customers Leave - Simple Analysis")
    
    # Convert Churn to numeric if needed
    if not pd.api.types.is_numeric_dtype(df['Churn']):
        df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
    
    # 1. Usage Patterns
    st.header("1. Usage Patterns")
    fig1 = px.box(df, x='Churn', y='ViewingHoursPerWeek',
                  title='Viewing Hours: Churned vs Retained Customers',
                  labels={'ViewingHoursPerWeek': 'Weekly Viewing Hours'})
    fig1.update_layout(xaxis_title="Customer Status (1 = Churned, 0 = Retained)")
    st.plotly_chart(fig1)
    
    # 2. Customer Satisfaction
    st.header("2. Customer Satisfaction")
    fig2 = px.histogram(df, x='UserRating', color='Churn',
                       title='User Ratings Distribution',
                       barmode='group',
                       labels={'UserRating': 'Rating Given by User'})
    st.plotly_chart(fig2)
    
    # 3. Support Issues
    st.header("3. Support Issues")
    fig3 = px.scatter(df, x='SupportTicketsPerMonth', y='Negative_Feedback',
                      color='Churn', 
                      title='Support Tickets vs Negative Feedback',
                      labels={
                          'SupportTicketsPerMonth': 'Monthly Support Tickets',
                          'Negative_Feedback': 'Negative Feedback Count'
                      })
    st.plotly_chart(fig3)
    
    # 4. Pricing Impact
    st.header("4. Pricing Impact")
    fig4 = px.box(df, x='Churn', y='MonthlyCharges',
                  title='Monthly Charges: Churned vs Retained Customers',
                  labels={'MonthlyCharges': 'Monthly Charges ($)'})
    fig4.update_layout(xaxis_title="Customer Status (1 = Churned, 0 = Retained)")
    st.plotly_chart(fig4)
    
    # 5. Key Risk Indicators
    st.header("5. Key Risk Indicators")
    risk_factors = [
        'Late_Payments',
        'Downgraded_Plan',
        'Auto_Renewal_Off',
        'Uses_Competitor_Platforms'
    ]
    
    risk_data = []
    for factor in risk_factors:
        churn_rate = df[df[factor] == 1]['Churn'].mean() * 100
        risk_data.append({
            'Factor': factor,
            'Churn Rate (%)': churn_rate
        })
    
    risk_df = pd.DataFrame(risk_data)
    fig5 = px.bar(risk_df, x='Factor', y='Churn Rate (%)',
                  title='Churn Rate by Risk Factor',
                  labels={'Factor': 'Risk Factor'})
    st.plotly_chart(fig5)
    
    # Summary Stats
    st.header("Summary of Key Findings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_viewing_churned = df[df['Churn'] == 1]['ViewingHoursPerWeek'].mean()
        avg_viewing_retained = df[df['Churn'] == 0]['ViewingHoursPerWeek'].mean()
        st.metric("Avg Weekly Viewing Hours", 
                 f"{avg_viewing_retained:.1f} hrs (retained)",
                 f"{avg_viewing_churned:.1f} hrs (churned)")
    
    with col2:
        avg_rating_churned = df[df['Churn'] == 1]['UserRating'].mean()
        avg_rating_retained = df[df['Churn'] == 0]['UserRating'].mean()
        st.metric("Avg User Rating", 
                 f"{avg_rating_retained:.1f} (retained)",
                 f"{avg_rating_churned:.1f} (churned)")
    
    with col3:
        avg_tickets_churned = df[df['Churn'] == 1]['SupportTicketsPerMonth'].mean()
        avg_tickets_retained = df[df['Churn'] == 0]['SupportTicketsPerMonth'].mean()
        st.metric("Avg Monthly Support Tickets", 
                 f"{avg_tickets_retained:.1f} (retained)",
                 f"{avg_tickets_churned:.1f} (churned)")

    # Clear Recommendations
    st.header("Clear Action Items")
    st.write("""
    Based on the data, here are the key areas to focus on:
    
    1. **Engagement Alert**: Customers watching less than average hours per week need attention
    2. **Support Focus**: High number of support tickets is a strong churn indicator
    3. **Pricing Sensitivity**: Review pricing for customers with higher monthly charges
    4. **Risk Monitoring**: Watch for late payments and plan downgrades
    5. **Competitor Analysis**: Users with competitor platforms are at higher risk
    """)

# --------------------------
# Main App
# --------------------------
def main():
    st.set_page_config(page_title="Customer Analytics Suite", layout="wide")
    data = load_data()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Analysis Type", 
                           ["Customer Clustering", 
                            "Churn Prediction", 
                            "Retention Strategies",
                            "Why Customers Leave?"])
    
    if page == "Customer Clustering":
        clustering_page(data)
    elif page == "Churn Prediction":
        prediction_page(data)
    elif page == "Retention Strategies":
        retention_strategies_page(data)
    # elif page == "Time-to-Churn Analysis":
    #     # Create a new page for time-to-churn analysis
    #     st.title("🔮 Time-to-Churn Forecast Dashboard")
    #     try:
    #         cleaned_df = clean_dataset(data)
    #         if cleaned_df is not None:
    #             prepared_data = prepare_data_for_survival(cleaned_df)
    #             cox_model = fit_cox_model(prepared_data)
    #             if cox_model is not None:
    #                 # Display summary statistics
    #                 st.subheader("📊 Model Summary")
    #                 col1, col2 = st.columns(2)
                    
    #                 with col1:
    #                     st.metric("Churn Rate", f"{(data['Churn'].value_counts()[1] / len(data) * 100):.1f}%")
                    
    #                 with col2:
    #                     st.metric("Median Account Age", f"{data['AccountAge'].median():.0f} days")
    #                     st.metric("Average Monthly Charges", f"${data['MonthlyCharges'].mean():.2f}")
                    
    #                 # Show survival curves
    #                 st.subheader("📈 Survival Analysis")
    #                 survival_fig = create_survival_curves_plot(cox_model, prepared_data)
    #                 st.plotly_chart(survival_fig, use_container_width=True)
                    
    #                 # Show risk factors
    #                 st.subheader("⚠️ Risk Factors")
    #                 risk_fig = create_risk_factors_plot(cox_model)
    #                 st.plotly_chart(risk_fig, use_container_width=True)
        
    #                 # Individual customer predictions
    #                 st.subheader("🎯 Individual Customer Predictions")
        
    #                 # Customer selector
    #                 selected_customer = st.selectbox(
    #                     "Select a customer to analyze:",
    #                     range(len(df)),
    #                     format_func=lambda x: f"Customer {x+1}"
    #                 )
        
    #                 # Show customer details
    #                 customer_data = cleaned_df.iloc[selected_customer]
    #                 col1, col2, col3 = st.columns(3)
        
    #                 with col1:
    #                     st.write("**Subscription Details:**")
    #                     st.write(f"Type: {customer_data['SubscriptionType']}")
    #                     st.write(f"Payment: {customer_data['PaymentMethod']}")
                        
    #                 with col2:
    #                     st.write("**Usage Metrics:**")
    #                     st.write(f"Monthly Charges: ${customer_data['MonthlyCharges']:.2f}")
    #                     st.write(f"Account Age: {customer_data['AccountAge']} days")
                        
    #                 with col3:
    #                     st.write("**Sentiment Metrics:**")
    #                     st.write(f"User Rating: {customer_data['UserRating']}/5")
    #                     st.write(f"Sentiment Score: {customer_data['Final_Sentiment_Score']:.2f}")
        
    #     # Calculate and show survival probability
    #                 customer_surv = cox_model.predict_survival_function(prepared_data.iloc[selected_customer:selected_customer+1])
        
    #     # Find the time at which survival probability drops below 0.5
    #                 median_survival = customer_surv.index[np.where(customer_surv.values < 0.5)[0][0]] if any(customer_surv.values < 0.5) else ">365"
        
    #                 st.metric(
    #                     "Predicted Days Until Churn Risk",
    #                     f"{median_survival} days",
    #                     delta=f"{int(median_survival) - customer_data['AccountAge']} days remaining" if isinstance(median_survival, (int, float)) else "Low risk"
    #                 )
        
    #     except Exception as e:
    #         st.error(f"Error processing data: {str(e)}")
    #         st.info("Please ensure your dataset contains all required columns and proper data formats.")
    else:
        analyze_churn_simple(data)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

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
# Retention Strategies Page Functions
# --------------------------
def retention_strategies_page(df):
    st.title("Retention Strategies Recommendations")
    
    st.markdown("""
    Our AI-driven model not only predicts churn but also recommends personalized retention actions to keep your customers engaged.
    """)

    # Ensure 'CustomerID' exists or create one
    if 'CustomerID' not in df.columns:
        st.warning("CustomerID not found. Creating a unique index-based identifier.")
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
# Main App
# --------------------------
def main():
    st.set_page_config(page_title="Customer Analytics Suite", layout="wide")
    data = load_data()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Analysis Type", 
                            ["Customer Clustering", "Churn Prediction", "Retention Strategies"])
    
    if page == "Customer Clustering":
        clustering_page(data)
    elif page == "Churn Prediction":
        prediction_page(data)
    elif page == "Retention Strategies":
        retention_strategies_page(data)

if __name__ == "__main__":
    main()


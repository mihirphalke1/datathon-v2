import pandas as pd
import streamlit as st

def calculate_clv(df):
    """Calculate Customer Lifetime Value (CLV) based on historical data."""
    df = df.copy()
    
    # Ensure required columns exist
    required_columns = {'TotalCharges', 'MonthlyCharges', 'AccountAge'}
    if not required_columns.issubset(df.columns):
        st.error(f"Missing required columns for CLV calculation: {required_columns - set(df.columns)}")
        return df

    # Handle missing values
    df.fillna({'TotalCharges': df['TotalCharges'].median(), 
               'MonthlyCharges': df['MonthlyCharges'].median(),
               'AccountAge': df['AccountAge'].median()}, inplace=True)

    # Average Purchase Value (APV)
    df['AveragePurchaseValue'] = df['TotalCharges'] / df['AccountAge']
    
    # Purchase Frequency (Assume AccountAge represents billing cycles)
    df['PurchaseFrequency'] = df['AccountAge']
    
    # Customer Lifespan (Assumed equal to AccountAge for now)
    df['CustomerLifespan'] = df['AccountAge']
    
    # Calculate CLV
    df['CLV'] = (df['AveragePurchaseValue'] * df['PurchaseFrequency']) * df['CustomerLifespan']
    
    return df

# ----------------------------------
# STREAMLIT DASHBOARD FOR CLV
# ----------------------------------
def main():
    st.title("📈 Customer Lifetime Value (CLV) Prediction")

    # Load dataset
    df = pd.read_csv("data/churn_predictions.csv")
        
        # Display raw data
    st.subheader("📌 Raw Data Sample")
    st.dataframe(df.head())

        # Calculate CLV
    df = calculate_clv(df)
        
        # Show updated data with CLV
    st.subheader("🔍 Data with CLV")
    st.dataframe(df[['CustomerID', 'CLV']].head() if 'CustomerID' in df.columns else df.head())

        # Save results
    df.to_csv("data/clv_predictions.csv", index=False)
    st.success("✅ CLV predictions saved to 'data/clv_predictions.csv'")

if __name__ == "__main__":
    main()
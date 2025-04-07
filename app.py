import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("C:/Workspace/datasets/train (House Price) CLEAN.csv")
    
    # Data cleaning steps from notebook
    data['BedroomAbvGr'].fillna(data['BedroomAbvGr'].median(), inplace=True)
    data = data[(data['BedroomAbvGr'] > 0) & (data['GrLivArea'] < 5000)]
    data = data[(data['SalePrice'] < 500000)]
    
    # Log transformations
    data['LogSalePrice'] = np.log1p(data['SalePrice'])
    data['LogGrLivArea'] = np.log1p(data['GrLivArea'])
    data['Logbedroomabvgr'] = np.log1p(data['BedroomAbvGr'])
    
    return data

# Train model
@st.cache_resource
def train_model(data):
    X = data[['Logbedroomabvgr', 'LogGrLivArea']]
    y = data['LogSalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    return model

# Main app
def main():
    st.title("House Price Prediction")
    st.write("Using XGBoost regression model")
    
    # Load data and model
    data = load_data()
    model = train_model(data)
    
    # Sidebar inputs
    st.sidebar.header("Input Features")
    bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 8, 3)
    liv_area = st.sidebar.slider("Living Area (sqft)", 500, 5000, 1500)
    
    # Predict on user input
    if st.sidebar.button("Predict Price"):
        # Create input DataFrame
        input_df = pd.DataFrame([[bedrooms, liv_area]],
                               columns=['BedroomAbvGr', 'GrLivArea'])
        
        # Apply log transforms
        input_df['Logbedroomabvgr'] = np.log1p(input_df['BedroomAbvGr'])
        input_df['LogGrLivArea'] = np.log1p(input_df['GrLivArea'])
        
        # Make prediction
        log_pred = model.predict(input_df[['Logbedroomabvgr', 'LogGrLivArea']])
        price_pred = np.expm1(log_pred)[0]
        
        st.subheader(f"Predicted House Price: ₹{price_pred:,.2f}")
    
    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.subheader("House Price Data")
        st.write(data)

if __name__ == "__main__":
    main()
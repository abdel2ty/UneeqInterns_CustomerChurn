import streamlit as st
import pandas as pd
import joblib

# Load the trained KNN model
model = joblib.load("knn_churn_model.pkl")

st.title("Customer Churn Prediction - Upload Test File")

st.write("Upload a CSV file with customer data to predict churn.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV
    test_data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(test_data.head())

    # Encode categorical columns like we did for training
    test_data['Gender_Female'] = test_data['Gender'].apply(lambda x: 1 if x=="Female" else 0)
    test_data['Subscription Type_Standard'] = test_data['Subscription Type'].apply(lambda x: 1 if x=="Standard" else 0)
    test_data['Contract Length_Quarterly'] = test_data['Contract Length'].apply(lambda x: 1 if x=="Quarterly" else 0)
    test_data['Contract Length_Annual'] = test_data['Contract Length'].apply(lambda x: 1 if x=="Annual" else 0)

    # Select the features for prediction
    features = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 
                'Total Spend', 'Last Interaction', 
                'Gender_Female', 'Subscription Type_Standard', 
                'Contract Length_Quarterly', 'Contract Length_Annual']
    
    X_test = test_data[features]

    # Predict churn
    predictions = model.predict(X_test)
    test_data['Churn_Prediction'] = predictions

    # Show predictions
    st.write("Predictions:")
    st.dataframe(test_data[['CustomerID', 'Churn_Prediction']])
    
    # Download predictions as CSV
    st.download_button(
        label="Download Predictions as CSV",
        data=test_data.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )
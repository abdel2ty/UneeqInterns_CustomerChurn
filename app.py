import streamlit as st
import pandas as pd
import joblib
import os 

# Use relative path to load the model (same folder as this app)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "knn_churn_model.pkl")
model = joblib.load(MODEL_PATH)

st.title("Customer Churn Prediction - Upload Test File")

st.write("Upload a CSV file with customer data to predict churn.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(test_data.head())

    # Encode categorical columns
    test_data['Gender_Female'] = test_data['Gender'].apply(lambda x: 1 if x=="Female" else 0)
    test_data['Subscription Type_Standard'] = test_data['Subscription Type'].apply(lambda x: 1 if x=="Standard" else 0)
    test_data['Contract Length_Quarterly'] = test_data['Contract Length'].apply(lambda x: 1 if x=="Quarterly" else 0)
    test_data['Contract Length_Annual'] = test_data['Contract Length'].apply(lambda x: 1 if x=="Annual" else 0)

    features = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 
                'Total Spend', 'Last Interaction', 
                'Gender_Female', 'Subscription Type_Standard', 
                'Contract Length_Quarterly', 'Contract Length_Annual']
    
    X_test = test_data[features]

    predictions = model.predict(X_test)
    test_data['Churn_Prediction'] = predictions

    st.write("Predictions:")
    st.dataframe(test_data[['CustomerID', 'Churn_Prediction']])
    
    st.download_button(
        label="Download Predictions as CSV",
        data=test_data.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

import streamlit as st
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
import joblib

MODEL_PATH = "knn_churn_model.pkl"

# Check if model exists, if not, train it
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.write("Training KNN model...")
    # هنا حط X_train و y_train جاهزين أو حملهم من ملف CSV
    train_data = pd.read_csv("train_data.csv")
    X_train = train_data.drop(columns=['Churn','CustomerID'])
    # Encode categorical columns
    X_train['Gender_Female'] = train_data['Gender'].apply(lambda x: 1 if x=="Female" else 0)
    X_train['Subscription Type_Standard'] = train_data['Subscription Type'].apply(lambda x: 1 if x=="Standard" else 0)
    X_train['Contract Length_Quarterly'] = train_data['Contract Length'].apply(lambda x: 1 if x=="Quarterly" else 0)
    X_train['Contract Length_Annual'] = train_data['Contract Length'].apply(lambda x: 1 if x=="Annual" else 0)
    X_train = X_train[['Age','Tenure','Usage Frequency','Support Calls','Payment Delay','Total Spend','Last Interaction',
                       'Gender_Female','Subscription Type_Standard','Contract Length_Quarterly','Contract Length_Annual']]
    
    y_train = train_data['Churn']
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    st.write("Model trained and saved!")

st.write("Model is ready to use!")
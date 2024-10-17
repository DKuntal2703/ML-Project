# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix

# Title and description
st.title("Faulty Steel Plate Prediction")
st.write("Upload your dataset to predict faulty steel plates.")

# File upload
url = "https://raw.githubusercontent.com/ML-Project/faults.csv"

@st.cache
def load_data():
    return pd.read_csv(url)

# Use online data if no file is uploaded
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset Preview:", df.head())
else:
    st.write("No file uploaded. Fetching sample data from online source.")
    df = load_data()
    st.write("Sample Data Preview:", df)

    # Target columns with multiple fault types
    target_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

    # Convert multi-label columns into a single target variable
    df['Fault_Type'] = df[target_columns].idxmax(axis=1)
    df = df.drop(columns=target_columns)

    # Splitting dataset
    X = df.drop(columns=['Fault_Type'])
    y = df['Fault_Type']

    # Data scaling
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

    # Model training
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Single prediction input
    st.write("Make a Prediction:")
    user_input = []
    for feature in df.columns[:-1]:  # Exclude 'Fault_Type'
        value = st.number_input(f"Enter value for {feature}:")
        user_input.append(value)

    if st.button("Predict"):
        input_scaled = scaler.transform([user_input])
        prediction = model.predict(input_scaled)
        st.write(f"The predicted fault type is: {prediction[0]}")
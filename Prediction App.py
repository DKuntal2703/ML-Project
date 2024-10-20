import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO

# Function to load model from a URL (e.g., GitHub raw file link)
@st.cache_resource
def load_model_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if request was successful
        model = pickle.load(BytesIO(response.content))  # Load model from response content
        return model
    except Exception as e:
        st.error(f"Error loading model from URL: {e}")
        st.stop()

# GitHub raw URL for the model file (update with your actual link)
MODEL_URL = 'https://github.com/DKuntal2703/ML-Project/blob/8a2ba0a1664a36b6dbf6bcc4ba44242b06efe404/model.pkl'

# Load model from GitHub URL
model = load_model_from_url(MODEL_URL)

# URL of the dataset hosted online
DATA_URL = "https://github.com/DKuntal2703/ML-Project/blob/8a2ba0a1664a36b6dbf6bcc4ba44242b06efe404/faults.csv"

# Function to load dataset from URL
@st.cache
def load_data():
    return pd.read_csv(DATA_URL)

# Title of the app
st.title('Steel Plate Fault Prediction')

# Description
st.markdown("""
This app uses machine learning to predict faults in steel plates.
Upload data to make predictions or use the default dataset provided.
""")

# File uploader for user data
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Load online data if no file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview (Uploaded File):")
else:
    df = load_data()
    st.write("Data Preview (Using Online Dataset):")

st.dataframe(df.head())

# Data Preprocessing (example)
if 'Steel_type' in df.columns:
    df['Steel_type'] = df['Steel_type'].astype('category').cat.codes

# Prepare the features (assuming 'Fault' is the target column)
X = df.drop(columns=['Fault'], errors='ignore')

# Perform prediction
predictions = model.predict(X)

# Display the predictions
df['Predicted_Fault'] = predictions
st.write("Predictions:")
st.dataframe(df[['Predicted_Fault']])

# Optionally download the results
csv = df.to_csv(index=False)
st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv")

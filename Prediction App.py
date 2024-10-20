import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Model loading with error handling
try:
    with open('xgb_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file not found. Please upload the correct model.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Load pre-trained model
# Assuming you have the model saved as 'xgb_model.pkl'
with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Title of the app
st.title('Steel Plate Fault Prediction')

# Description
st.markdown("""
This app uses machine learning to predict faults in steel plates.
Upload data to make predictions or test with the example input provided.
""")

# Upload file section
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Data Preview:")
    st.dataframe(df.head())

    # Data Preprocessing
    label_encoder = LabelEncoder()

    # Encode Steel_type column
    df['Steel_type'] = label_encoder.fit_transform(df['Steel_type'])

    # Prepare the features (assuming you know the required columns)
    X = df.drop(['Fault'], axis=1)  # Drop the target column 'Fault'

    # Perform prediction
    predictions = model.predict(X)
    
    # Display the predictions
    df['Predicted_Fault'] = predictions
    st.write("Predictions:")
    st.dataframe(df[['Predicted_Fault']])

    # Optionally download the results
    csv = df.to_csv(index=False)
    st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv")
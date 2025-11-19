import streamlit as st
import requests
import numpy as np

st.title("EEG Attention State Predictor")

st.write("Upload EEG data (3584 values: 256 samples x 14 channels) or enter manually.")

# Option to upload file or enter manually
option = st.radio("Choose input method:", ("Upload CSV", "Manual Input"))

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = np.loadtxt(uploaded_file, delimiter=',')
        if data.shape[0] != 3584:
            st.error("CSV must contain exactly 3584 values.")
        else:
            eeg_data = data.tolist()
            st.success("Data loaded successfully!")
else:
    st.write("Enter 3584 EEG values (comma-separated):")
    input_text = st.text_area("EEG Data", height=100)
    if input_text:
        try:
            eeg_data = [float(x.strip()) for x in input_text.split(',')]
            if len(eeg_data) != 3584:
                st.error("Must enter exactly 3584 values.")
            else:
                st.success("Data entered successfully!")
        except ValueError:
            st.error("Invalid input. Please enter numbers separated by commas.")

# Predict button
if st.button("Predict Attention State"):
    if 'eeg_data' in locals():
        # Assuming API is running on localhost:8000
        response = requests.post("http://localhost:8000/predict", json={"eeg_data": eeg_data})
        if response.status_code == 200:
            result = response.json()
            st.write(f"Predicted State: {result['prediction']}")
        else:
            st.error("Error in prediction. Make sure the API is running.")
    else:
        st.error("Please provide EEG data first.")

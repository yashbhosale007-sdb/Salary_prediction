import streamlit as st
import pickle
import numpy as np

# Page configuration for a "Good Front End" look
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

# Custom CSS to spruce up the UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    with open('model.3.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the model
model = load_model()

# Header Section
st.title("💰 Salary Estimation Tool")
st.write("Enter your years of professional experience to estimate your projected salary based on our ML model.")

st.divider()

# Input Section
col1, col2 = st.columns([2, 1])

with col1:
    years_exp = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    predict_btn = st.button("Predict Salary")

# Prediction Logic
if predict_btn:
    # Reshape input for sklearn (expects 2D array)
    input_data = np.array([[years_exp]])
    prediction = model.predict(input_data)
    
    st.balloons()
    
    # Display Result
    st.success(f"### Estimated Salary: ${prediction[0]:,.2f}")
    
    # Adding some context
    st.info(f"This prediction is based on a Linear Regression model trained on your 'YearsExperience' dataset.")

# Footer
st.divider()
st.caption("Model Version: Scikit-Learn 1.6.1 | Deployment: Streamlit")

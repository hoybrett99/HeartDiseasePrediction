import streamlit as st
from PIL import Image



st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)  # Adjust width as needed

def home_page():
    st.title("Heart Disease Prediction App")

    st.write("""
    Welcome to the Heart Disease Prediction App! This application is designed to help you assess your risk of heart disease based on various personal health indicators. 

    ## Features
    - **Heart Disease Prediction**: Utilize a robust Random Forest algorithm to predict the likelihood of heart disease based on your health data.
    - **BMI Calculator**: Easily calculate your Body Mass Index (BMI) if you don't already know it, which is an important factor in heart disease risk assessment.
    - **Data Dashboard**: Gain insights from the dataset through visualizations that summarize key health indicators related to heart disease.

    ## How It Works
    1. **Input Your Information**: Fill in the details about your health and lifestyle in the sidebar.
    2. **Get Your Prediction**: After entering your information, click the "Calculate" button to see your risk of heart disease.
    3. **Understand Your Results**: View the probability of heart disease and explore additional insights through the dashboard.

    ###### For more information, check the about section""")

# Call this function in your main app
home_page()
import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="❔"
)

st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)  # Adjust width as needed

def about_section():
    st.title("About This Heart Disease Prediction App")

    st.write("""
    ## Overview
    This application is designed to predict the likelihood of heart disease based on various personal health indicators. The prediction model is built using a Random Forest algorithm, which has shown robust performance in classification tasks. In addition to the prediction feature, the app includes:

    1. **BMI Calculator**: This tool helps users calculate their Body Mass Index (BMI) if they don't know it, ensuring accurate input for the prediction model.
    2. **Dashboard**: This feature provides a summary of the dataset, offering users insights into the overall distribution and trends of heart disease indicators.

    ## Dataset
    The dataset used for this project is the [Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data) from Kaggle. This dataset was collected as part of the 2020 annual Behavioral Risk Factor Surveillance System (BRFSS) survey, which is conducted by the Centers for Disease Control and Prevention (CDC).

    ## Model Performance
    My Random Forest model achieved:
    - 77% accuracy in predicting people with heart disease
    - 75% accuracy in predicting people without heart disease

    These results demonstrate the model's balanced performance in identifying both positive and negative cases of heart disease.

    ## About the Developer
    I am a data enthusiast with a huge passion for:
    - Building machine learning models
    - Creating Streamlit web applications
    - Web scraping and data collection

    My work focuses on leveraging data science and machine learning techniques to create practical, user-friendly applications that can provide valuable insights and predictions.

    ### Connect with Me
    Feel free to check out my other projects and connect with me on GitHub:
    [https://github.com/hoybrett99](https://github.com/hoybrett99)

    I'm open to collaboration and always eager to learn from fellow data enthusiasts and professionals in the field.
             
    ### Share the Code
    You can find the source code for this project on GitHub:
    [Heart Disease Prediction GitHub Repository](https://github.com/hoybrett99/HeartDiseasePrediction)
    """)

# Call this function in your main app
about_section()
import pandas as pd
import pickle
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from PIL import Image
import gzip
import requests
import io


st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)  # Adjust width as needed

# Main header of the app
st.write("""
        # Heart Disease Predictor
         """)

# Main header of the side bar
st.sidebar.header("Please fill in the information to find your result")


def user_input_features():
        Sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
        GeneralHealth = st.sidebar.selectbox("How's your overall health?", ("Excellent", "Very good", "Good", "Fair", "Poor"))
        PhysicalHealthDays = st.sidebar.slider("How many days were you sick/injured over the last 30 days?", 0, 30, 10)
        MentalHealthDays = st.sidebar.slider("How many days during the past 30 days was your mental health not good?", 0, 30, 10)
        LastCheckupTime = st.sidebar.selectbox("Last Checkup?", ("< 1 year", "< 2 years", "< 5 years", "5+ years"))
        PhysicalActivities = st.sidebar.selectbox("Are you physically active?", ("No", "Yes")),
        SleepHours = st.sidebar.slider("How long do you sleep each day?", 1.0, 24.0, 8.0)
        RemovedTeeth = st.sidebar.selectbox("How many teeth do you have removed?", ("0", "1-5", "6+", "All"))
        HadStroke = st.sidebar.selectbox("Have you ever had a stroke?", ("No", "Yes")),
        HadAsthma = st.sidebar.selectbox("Have you ever had asthma?", ("No", "Yes")),
        HadSkinCancer = st.sidebar.selectbox("Have you ever had skin cancer?", ("No", "Yes")),
        HadCOPD = st.sidebar.selectbox("Have you ever had Chronic Obstructive Pulmonary Disease?", ("No", "Yes")),
        HadDepressiveDisorder = st.sidebar.selectbox("Have you ever had depressive disorder?", ("No", "Yes")),
        HadKidneyDisease = st.sidebar.selectbox("Have you ever had kidney disease?", ("No", "Yes")),
        HadArthritis = st.sidebar.selectbox("Have you ever had arthritis?", ("No", "Yes")),
        DeafOrHardOfHearing = st.sidebar.selectbox("Do you have any difficulty hearing?", ("No", "Yes")),
        BlindOrVisionDifficulty = st.sidebar.selectbox("Do you have any difficulty with your vision?", ("No", "Yes")),
        DifficultyConcentrating = st.sidebar.selectbox("Are you have any difficulties concentrating?", ("No", "Yes")),
        DifficultyWalking = st.sidebar.selectbox("Are you struggling to walk?", ("No", "Yes")),
        DifficultyDressingBathing = st.sidebar.selectbox("Do you have any difficulty dressing or bathing?", ("No", "Yes")),
        DifficultyErrands = st.sidebar.selectbox("Do you have any difficulty running errands?", ("No", "Yes")),
        ChestScan = st.sidebar.selectbox("Have you had a chest scan?", ("No", "Yes")),
        AgeCategory = st.sidebar.selectbox("What is your age category?", ("18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"))
        HeightInMeters = st.sidebar.number_input("What is your height in meters?", min_value=0.0, format="%.2f"),
        BMI = st.sidebar.number_input("What is your BMI?", min_value=0.0, format="%.2f"),
        AlcoholDrinkers = st.sidebar.selectbox("Do you consume alcohol?", ("No", "Yes")),
        HIVTesting = st.sidebar.selectbox("Have you ever been tested for HIV?", ("No", "Yes")),
        FluVaxLast12 = st.sidebar.selectbox("Have you received a flu vaccine in the last 12 months?", ("No", "Yes")),
        PneumoVaxEver = st.sidebar.selectbox("Have you ever received a pneumococcal vaccine?", ("No", "Yes")),
        HighRiskLastYear = st.sidebar.selectbox("Have you been in a high-risk category in the last year?", ("No", "Yes")),
        HadDiabetes = st.sidebar.selectbox("Do you have diabetes?", ("No", "Borderline (Pre-Diabetes)", "Yes", "Yes, during pregnancy"))
        SmokerStatus = st.sidebar.selectbox("What is your smoking status?", ("Never", "Previously", "Sometimes", "Everyday"))
        ECigaretteUsage = st.sidebar.selectbox("What is your e-cigarette usage?", ("Never", "Recently", "Sometimes", "Everyday"))
        RaceEthnicityCategory = st.sidebar.selectbox("What is your race/ethnicity?", ("White only, Non-Hispanic", "Black only, Non-Hispanic", "Hispanic", "Multiracial, Non-Hispanic", "Other race only, Non-Hispanic"))
        TetanusLast10Tdap = st.sidebar.selectbox("Have you had a tetanus vaccine in the last 10 years?", ("No", "Yes", "Yes, but not Tdap", "Yes, but not sure what type"))
        CovidPos = st.sidebar.selectbox("Have you tested positive for COVID-19?", ("No", "Yes", "Tested positive using home test without a health professional")),
        data = {
        "Sex": Sex,
        "GeneralHealth": GeneralHealth,
        "PhysicalHealthDays": PhysicalHealthDays,
        "MentalHealthDays": MentalHealthDays,
        "LastCheckupTime": LastCheckupTime,
        "PhysicalActivities": PhysicalActivities,
        "SleepHours": SleepHours,
        "RemovedTeeth": RemovedTeeth,
        "HadStroke": HadStroke,
        "HadAsthma": HadAsthma,
        "HadSkinCancer": HadSkinCancer,
        "HadCOPD": HadCOPD,
        "HadDepressiveDisorder": HadDepressiveDisorder,
        "HadKidneyDisease": HadKidneyDisease,
        "HadArthritis": HadArthritis,
        "DeafOrHardOfHearing": DeafOrHardOfHearing,
        "BlindOrVisionDifficulty": BlindOrVisionDifficulty,
        "DifficultyConcentrating": DifficultyConcentrating,
        "DifficultyWalking": DifficultyWalking,
        "DifficultyDressingBathing": DifficultyDressingBathing,
        "DifficultyErrands": DifficultyErrands,
        "ChestScan": ChestScan,
        "AgeCategory": AgeCategory,
        "HeightInMeters": HeightInMeters,
        "BMI": BMI,
        "AlcoholDrinkers": AlcoholDrinkers,
        "HIVTesting": HIVTesting,
        "FluVaxLast12": FluVaxLast12,
        "PneumoVaxEver": PneumoVaxEver,
        "HighRiskLastYear": HighRiskLastYear,
        "HadDiabetes": HadDiabetes,
        "SmokerStatus": SmokerStatus,
        "ECigaretteUsage": ECigaretteUsage,
        "RaceEthnicityCategory": RaceEthnicityCategory,
        "TetanusLast10Tdap": TetanusLast10Tdap,
        "CovidPos": CovidPos
    }
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()



# Importing the dataset
heart = pd.read_csv("https://raw.githubusercontent.com/hoybrett99/HeartDiseasePrediction/main/Sampled_heartdisease.csv")
heart = heart.drop("HeartDisease", axis=1)

# Copying the dataset
df = pd.concat([input_df, heart], axis=0)

# Encoding binary variables
binary_columns = df.columns[df.nunique() == 2]

# Creating a map
binary_mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}

# Replacing the values
for col in binary_columns:
    df[col] = df[col].replace(binary_mapping)


# Define the mapping dictionary
checkup_mapping = {
    '< 1 year': 0,
    '< 2 years': 1,
    '< 5 years': 2,
    '5+ years': 3
}

# Apply the mapping using replace
df['LastCheckupTime'] = df['LastCheckupTime'].replace(checkup_mapping)

# Define the mapping dictionary
teeth_mapping = {
    '0': 0,
    '"1-5"': 1,
    '6+': 2,
    'All': 3
}

# Apply the mapping using replace
df['RemovedTeeth'] = df['RemovedTeeth'].replace(teeth_mapping)

# Mapping dictionary
general_health_mapping = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very good': 3,
    'Excellent': 4
}

# Apply the mapping using replace
df['GeneralHealth'] = df['GeneralHealth'].replace(general_health_mapping)


# Mapping dictionary
age_mapping = {
    '65-69': 9,
    '60-64': 8,
    '70-74': 10,
    '55-59': 7,
    '50-54': 6,
    '75-79': 11,
    '80+': 12,
    '40-44': 4,
    '45-49': 5,
    '35-39': 3,
    '30-34': 2,
    '18-24': 0,
    '25-29': 1
}



# Apply the mapping using replace
df['AgeCategory'] = df['AgeCategory'].replace(age_mapping)

# One-hot encoding the remaining categorical variables
categorical_col = df.select_dtypes(include="object").columns




for col in categorical_col:
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(col, axis=1)



st.subheader("User Input")
st.write(df.head(1))

# Extracting first row
df = df[:1]


# Function to download the model file from a URL
@st.cache_data
def download_model(url):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    return response.content

# URL of the gzipped pickle file
model_url = 'https://github.com/hoybrett99/HeartDiseasePrediction/raw/main/heart_rf.pkl.gz'

# Download the model
model_content = download_model(model_url)

# Load the model from the downloaded content
with gzip.open(io.BytesIO(model_content), 'rb') as f:
    rf = pickle.load(f)

# Button to get prediction
if st.sidebar.button("Get Prediction"):
    # Preprocess user inputs
    # You need to encode categorical variables properly
    # This is just a placeholder; replace with actual encoding logic
    user_data_encoded = df  # Replace with actual encoding logic

    # Get prediction
    rf_pred = rf.predict(user_data_encoded)
    rf_proba = rf.predict_proba(user_data_encoded)

    # Display prediction in the main area
    st.subheader("Prediction Result")

    # Create two columns for the layout
    col1, col2 = st.columns(2)  # This creates two equal-width columns

    # Probability KPIs
    no_prob = rf_proba[0][0]
    yes_prob = rf_proba[0][1]

    # Donut chart in the first column
    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=['None', 'Heart Disease'],
            values=[no_prob, yes_prob],
            hole=.3,  # This makes it a donut chart
            marker=dict(colors=['#0E1117', '#ff4b4b']),  # Optional: Custom colors
            textinfo='label+percent',
            textfont=dict(size=16),  # Increase text size
            insidetextfont=dict(size=14),  # Increase inside text size
            sort=False  # Preserve the order of labels
        )])

        fig.update_layout(
            showlegend=False,  # This removes the legend
            margin=dict(t=0, b=0, l=0, r=0),
            height=400,  # Adjust the height of the chart
            width=400   # Adjust the width of the chart
        )

        st.plotly_chart(fig)

    # Large KPI in the second column
    with col2:
        heart_disease_probability = rf_proba[0][1]  # Assuming the second value is for heart disease
        percentage = f"{heart_disease_probability:.1%}"
    
        # Define groups based on probability
        if heart_disease_probability <= 0.20:
            prediction = "Healthy"
            emoji = "🥳"
            color = "green"
        elif heart_disease_probability <= 0.40:
            prediction = "You are fine"
            emoji = "👍"
            color = "lightgreen"
        elif heart_disease_probability <= 0.60:
            prediction = "Consider seeing a doctor"
            emoji = "🤔"
            color = "orange"
        elif heart_disease_probability <= 0.80:
            prediction = "You should see a doctor"
            emoji = "⚠️"
            color = "orangered"
        else:
            prediction = "Definitely see a doctor"
            emoji = "🚨"
            color = "red"
    
        # Custom HTML to make the KPI larger
        html_content = f"""
        <div style="
            height: 400px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background-color: #0E1117;
            border-radius: 10px;
            padding: 20px;
        ">
            <h2 style="font-size: 48px; margin-bottom: 10px; color: white;">Prediction</h2>
            <div style="font-size: 28px; font-weight: bold; color: {color}; margin-bottom: 10px;">{prediction} {emoji}</div>
            <div style="font-size: 20px; color: white;">Probability of Heart Disease: <span style="color: {color};">{percentage}</span></div>
        </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)


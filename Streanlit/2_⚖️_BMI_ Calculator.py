import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats



def calculate_bmi(weight, height):
    """Calculate BMI given weight in kg and height in meters."""
    return weight / (height ** 2)

st.set_page_config(page_title="BMI Calculator", layout="wide")

st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)  # Adjust width as needed

st.title("BMI Calculator")

# Input fields for weight and height
weight = st.number_input("Weight (kg)", min_value=1.0, max_value=500.0, value=70.0, step=0.1)
height = st.number_input("Height (m)", min_value=0.1, max_value=3.0, value=1.70, step=0.01)

if st.button("Calculate BMI"):
    bmi = calculate_bmi(weight, height)
    st.markdown(f"<h1 style='text-align: center; color: #FFFFFF;'>Your BMI is: {bmi:.2f}</h1>", unsafe_allow_html=True)

    # Generate some sample BMI data for the KDE plot
    bmi_data = np.random.normal(loc=24, scale=4, size=1000)  # Sample data for demonstration

    # Calculate kernel density estimation
    kde = stats.gaussian_kde(bmi_data)
    x_range = np.linspace(bmi_data.min(), bmi_data.max(), 100)
    y_range = kde(x_range)

    # Create the line chart
    fig_bmi = go.Figure()
    fig_bmi.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', fill='tozeroy', line_color='#ff4b4b'))

    # Add a dot for the user's BMI
    fig_bmi.add_trace(go.Scatter(
        x=[bmi],
        y=[kde(bmi)[0]],
        mode='markers+text',
        text=[f'You are here!'],
        textposition='top center',
        marker=dict(color='#ff4b4b', size=10),
        textfont=dict(size=16)  # Increase text size for the BMI label
    ))

    fig_bmi.update_layout(
        title="BMI Distribution",
        xaxis_title="BMI",
        yaxis_title="Density",
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='white'),
        showlegend=False  # Remove the legend
    )

    st.plotly_chart(fig_bmi)
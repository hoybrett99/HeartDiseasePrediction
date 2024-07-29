
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st
from scipy import stats


# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Heart_disease/heart_2022_no_nans.csv")
    return df

df = load_data()

st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)  # Adjust width as needed

# Calculate heart disease percentage
df['HeartDisease'] = (df['HadHeartAttack'] == 'Yes') | (df['HadAngina'] == 'Yes')
heart_disease_percentage = (df['HeartDisease'].sum() / len(df)) * 100

# Sidebar for filters
st.sidebar.header("Filters")

# Sex Filter
gender_options = ['All'] + list(df['Sex'].unique())
gender = st.sidebar.selectbox("Select Gender", gender_options)

# Age Category Filter
age_categories = [
    "Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39",
    "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
    "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
    "Age 80 or older"
]

st.sidebar.subheader("Choose age:")
selected_age_categories = []
for age in age_categories:
    if st.sidebar.checkbox(age, value=True):
        selected_age_categories.append(age)

# Hours of Sleep Filter
sleep_hours = st.sidebar.slider("Select Hours of Sleep", 0, 24, (0, 24))

# Smoking Status Filter
st.sidebar.subheader("Smoking Status")
smoking_status = []
for status in df['SmokerStatus'].unique():
    if st.sidebar.checkbox(status, value=True):
        smoking_status.append(status)

# Diabetes Filter
st.sidebar.subheader("Diabetes Status")
diabetes_status = []
for status in df['HadDiabetes'].unique():
    if st.sidebar.checkbox(status, value=True):
        diabetes_status.append(status)

# Alcohol Consumption Filter
alcohol_options = ['All'] + list(df['AlcoholDrinkers'].unique())
alcohol_consumption = st.sidebar.selectbox("Select Alcohol Consumption", alcohol_options)

# Physical Activities Filter
physical_options = ['All'] + list(df['PhysicalActivities'].unique())
physical_activities = st.sidebar.selectbox("Select Physical Activity", physical_options)

# Filter the DataFrame based on selected filters
filtered_df = df[
    (df['AgeCategory'].isin(selected_age_categories)) &
    (df['SleepHours'].between(sleep_hours[0], sleep_hours[1])) &
    (df['SmokerStatus'].isin(smoking_status)) &
    (df['HadDiabetes'].isin(diabetes_status))
]

if alcohol_consumption != 'All':
    filtered_df = filtered_df[filtered_df['AlcoholDrinkers'] == alcohol_consumption]

if physical_activities != 'All':
    filtered_df = filtered_df[filtered_df['PhysicalActivities'] == physical_activities]

if gender != 'All':
    filtered_df = filtered_df[filtered_df['Sex'] == gender]

# Calculate additional KPIs
smokers_with_heart_disease_percentage = (filtered_df[(filtered_df['HeartDisease']) & (filtered_df['SmokerStatus'] != 'Never')].shape[0] / filtered_df[filtered_df['SmokerStatus'] != 'Never'].shape[0]) * 100

# Dashboard title
st.title("Heart Disease Dashboard")

# KPIs
col1, col2 = st.columns(2)
with col1:
    total_people_html = f"""
    <div style="text-align: center; color: white; background-color: #ff4b4b; padding: 10px; border-radius: 10px;">
        <p style="font-size: 24px; margin: 0;"><strong>No. of People Sampled</strong></p>
        <p style="font-size: 36px; margin: 0;">{len(filtered_df):,}</p>
    </div>
    """
    st.markdown(total_people_html, unsafe_allow_html=True)

with col2:
    heart_disease_percentage = (filtered_df['HeartDisease'].sum() / len(filtered_df)) * 100
    heart_disease_html = f"""
    <div style="text-align: center; color: white; background-color: #ff4b4b; padding: 10px; border-radius: 10px;">
        <p style="font-size: 24px; margin: 0;"><strong>% of People with Heart Disease</strong></p>
        <p style="font-size: 36px; margin: 0;">{heart_disease_percentage:.2f}%</p>
    </div>
    """
    st.markdown(heart_disease_html, unsafe_allow_html=True)

# State name to abbreviation mapping
state_abbr = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
    'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Convert state names to abbreviations and filter out territories
filtered_df['State'] = filtered_df['State'].map(state_abbr)
filtered_df = filtered_df[filtered_df['State'].notna()]  # Remove rows where State is not in our abbreviation list

# Calculate heart disease counts by state
state_counts = filtered_df[filtered_df['HeartDisease']].groupby('State').size().reset_index(name='Count')

# Create the choropleth map
fig_map = px.choropleth(state_counts,
                        locations='State', 
                        locationmode="USA-states", 
                        color='Count',
                        scope="usa",
                        color_continuous_scale="Reds")
fig_map.update_layout(
    title={
        'text': 'Heart Disease Cases by State',
        'yanchor': 'top',
        'font': {'size': 24}  # Increase the size of the title text
    },
    geo=dict(bgcolor='#0E1117'),  # Set the background color of the map
    paper_bgcolor='#0E1117',  # Set the background color of the entire figure
    plot_bgcolor='#0E1117',  # Set the background color of the plot area
    font=dict(color='white')  # Set the font color to white for better contrast
)

# Display the map in Streamlit
st.plotly_chart(fig_map)

# Age Category Bar Chart (Unfiltered)
age_data = df.groupby('AgeCategory')['HeartDisease'].mean().reset_index()
age_data['Percentage'] = age_data['HeartDisease'] * 100

# Remove "Age " prefix from AgeCategory labels
age_data['AgeCategory'] = age_data['AgeCategory'].str.replace('Age ', '')

# Sort the data for better visualization
age_data = age_data.sort_values('Percentage', ascending=True)

fig_age = px.bar(age_data, 
                 x='AgeCategory', 
                 y='Percentage',
                 title='Heart Disease Percentage by Age Category',
                 color='Percentage',
                 color_continuous_scale='Reds')

fig_age.update_layout(
    xaxis_title='Age Category', 
    yaxis_title='Percentage',
    paper_bgcolor='#0E1117',
    plot_bgcolor='#0E1117',
    font=dict(color='white'),
    showlegend=False,  # Remove the legend
    title=dict(
        text='% of Heart Disease by Age Category',
        font=dict(size=24)  # Set the title font size to 24
    )
)

fig_age.update_traces(marker_line_color='#0E1117', marker_line_width=1.5)

# Sex Donut Chart
sex_data = filtered_df.groupby('Sex')['HeartDisease'].mean().reset_index()
sex_data['Percentage'] = sex_data['HeartDisease'] * 100

# Define colors
female_color = '#ffbfbf'
male_color = '#ba0202'  # A darker shade of red

fig_sex = go.Figure(data=[go.Pie(
    labels=sex_data['Sex'],
    values=sex_data['Percentage'],
    hole=.3,
    marker=dict(colors=[female_color if sex == 'Female' else male_color for sex in sex_data['Sex']]),
    textinfo='label+percent',
    textposition='inside',
    insidetextorientation='radial',
    showlegend=False
)])

fig_sex.update_layout(
    title_text='Heart Disease Percentage by Sex',
    paper_bgcolor='#0E1117',
    plot_bgcolor='#0E1117',
    font=dict(color='white'),
    title=dict(
        text='% of Heart Disease by Gender',
        font=dict(size=24)  # Set the title font size to 24
))

# BMI Line Chart for people with heart disease
bmi_data = filtered_df[filtered_df['HeartDisease']]['BMI']

# Calculate kernel density estimation
kde = stats.gaussian_kde(bmi_data)
x_range = np.linspace(bmi_data.min(), bmi_data.max(), 100)
y_range = kde(x_range)

# Create the line chart
fig_bmi = go.Figure()
fig_bmi.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', fill='tozeroy', line_color='#ff4b4b'))

fig_bmi.update_layout(
    title={
        'text': 'BMI Distribution for People with Heart Disease',
        'font': {'size': 24},  # Set the title font size to 24
        'x': 0.5,  # Center the title horizontally
        'xanchor': 'center',
        'y': 0.95  # Adjust the vertical position of the title
    },
    xaxis_title="BMI",
    yaxis_title="Density",
    paper_bgcolor='#0E1117',
    plot_bgcolor='#0E1117',
    font=dict(color='white')
)

# Display charts with adjusted ratios
col1, col2 = st.columns([2, 1])  # 2/3 for bar chart, 1/3 for donut chart

with col1:
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    st.plotly_chart(fig_sex, use_container_width=True)

# Display the BMI line chart
st.plotly_chart(fig_bmi, use_container_width=True)
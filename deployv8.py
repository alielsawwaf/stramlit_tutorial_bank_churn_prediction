
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import joblib

# Set up the page title and theme (in the Streamlit settings file or code)
st.set_page_config(page_title="Bank Churn Prediction", page_icon="🔮", layout="wide")

# Custom title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Bank Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #555;'>Predicting Bank Customer Churn using Machine Learning</h2>", unsafe_allow_html=True)

# Descriptive introduction with Markdown
st.markdown("""
    <div style="text-align: center; font-size: 18px;">
    This tool uses machine learning to predict whether a customer will churn based on their information. 
    Enter your details to predict the likelihood of churn!
    </div>
""", unsafe_allow_html=True)

# Adding a background image (optional)
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://path_to_your_image.jpg');
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
        }
    </style>
""", unsafe_allow_html=True)

# Load pre-trained model and other assets
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Add an image with caption
    st.image("multimedia/5-1.png", caption="I'll help you diagnose chances of having Bank Churn!", width=300)

with col2:
    # Add team details in an elegant manner
    st.markdown("### **Team of Work**", unsafe_allow_html=True)
    st.write("***Eng. Ali Mohammed Mohammed Elsawwaf***")
    st.write("***Eng. Tamer Hosny Abd El Halim***")
    st.write("***Eng. Nadia***")
    st.write("***Eng. John Magdy Hanna***")

# Add a section with text describing how ML models help predict churn
with col1:
    st.markdown("""
        **Did you know that machine learning models can help you predict the chances of having Bank Churn?**
        In this app, you can estimate your chance of having Bank Churn (yes/no) in seconds!
    """)

with col2:
    st.image("multimedia/4.jfif", caption="Churn Prediction Model", width=300)

# User input form (sidebar)
st.sidebar.header("Customer Information")
import streamlit as st

############################
# # Set up sidebar navigation with hyperlinks
# st.sidebar.markdown("[🚀 Main Menu](http://www.google.com)")
# st.sidebar.markdown("""
# - [🏠 Home](?page=home)
# - [📝 Data Input](?page=input)
# - [📊 Visualization](?page=viz)
# - [ℹ️ About](?page=about)
# """)


################################

# Initialize an empty DataFrame to store user inputs
data = pd.DataFrame(columns=["Age", "Geography_Germany", "Geography_Spain", "Geography_France", "Gender", "IsActiveMember", "Balance", "EstimatedSalary"])

age = st.sidebar.number_input("Age in Years", 1, 100, 30, 1)
geography = st.sidebar.radio('Pick your Geography', ['Germany', 'Spain', 'France'])

# Encoding the geography selection
geography_Germany = 0
geography_Spain = 0
geography_France = 0
if geography == "Germany":
    geography_Germany = 1
elif geography == "Spain":
    geography_Spain = 1
elif geography == "France":
    geography_France = 1

gender = st.sidebar.radio('Pick your gender', ['Male', 'Female'])
gender = 1 if gender == "Male" else 0

isActiveMember = st.sidebar.radio('Active Member', ['Yes', 'No'])
isActiveMember = 1 if isActiveMember == "Yes" else 0

balance = st.sidebar.slider("Balance", 1.0, 300000.0, 10000.0, 100.0)
estimatedSalary = st.sidebar.slider("Estimated Salary", 1, 200000, 10000, 500)

creditScore = st.sidebar.number_input("CreditScore", 1, 100, 10, 1)
numOfProducts = st.sidebar.number_input("NumOfProducts", 1, 4, 1, 1)
tenure = st.sidebar.number_input("Tenure", 0, 11, 1, 1)

# Display form when the submit button is clicked
if st.sidebar.button("Submit"):
    # Collecting the user input data
    new_row = pd.DataFrame({
        "CreditScore": [creditScore],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [numOfProducts],
        "IsActiveMember": [isActiveMember],
        "EstimatedSalary": [estimatedSalary],
        "Geography_Germany": [geography_Germany],
        "Geography_Spain": [geography_Spain],
        "Geography_France": [geography_France]
    })

    # Show the entered data
    st.subheader("Your Submitted Data:")
    st.dataframe(new_row)

    # Scaling the numeric features using the preloaded scaler
    scale_vars = ['CreditScore', 'EstimatedSalary', 'Tenure', 'Balance', 'Age', 'NumOfProducts']
    new_row[scale_vars] = scaler.transform(new_row[scale_vars])

    # Show the scaled data
    st.subheader("Scaled Data:")
    st.dataframe(new_row)

    # Arranged scaled data
    features = ['Age', 'Geography_Germany', 'Geography_Spain', 'Geography_France',
                'Gender', 'IsActiveMember', 'Balance', 'EstimatedSalary']
    scaled_df = new_row[features]
    
    st.subheader("Scaled and Arranged Data:")
    st.dataframe(scaled_df)

    # Prediction from the model
    prediction = model.predict(scaled_df)
    prediction_text = "Churn Likelihood: **YES**" if prediction[0] == 1 else "Churn Likelihood: **NO**"
    st.markdown(f"<h2 style='text-align: center; color: #FF6347;'>{prediction_text}</h2>", unsafe_allow_html=True)

# Footer message
st.markdown("""
    <footer style="text-align: center; font-size: 14px; color: #888;">
        <p>Developed by <b>Team Bank Churn Prediction</b> | Machine Learning Model for Customer Churn Prediction </p>
        <b>DEPI 2025 IBM Data Scientist - CLS CAI2_AIS4_G4 - Round 2</b>
    </footer>
""", unsafe_allow_html=True)

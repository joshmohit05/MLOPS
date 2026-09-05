"""
Streamlit app: collects customer details from a form and predicts whether
they are likely to purchase the Wellness Tourism Package.

Runs directly from the repo on Streamlit Community Cloud, loading the
model file committed by the training step of the GitHub Actions pipeline.
"""

import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧘")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Wellness Tourism Package — Purchase Predictor")
st.write(
    "Enter a customer's details to estimate the likelihood they will "
    "purchase the newly introduced Wellness Tourism Package."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=180, value=15)
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

with col2:
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
    passport = st.selectbox("Holds Passport", ["No", "Yes"])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns a Car", ["No", "Yes"])
    number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0, value=22000)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Likely to purchase the Wellness Package (probability: {probability:.1%})")
    else:
        st.info(f"Unlikely to purchase the Wellness Package (probability: {probability:.1%})")

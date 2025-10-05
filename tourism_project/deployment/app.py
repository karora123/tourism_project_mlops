import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="karora1804/tourism-project-model", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package buy Prediction App")
st.write("""
The application estimates the probability of a customer purchasing a tourism package based on key input parameters. 
Kindly provide the requested customer details to receive a personalized prediction.
""")

# User input
age = st.number_input("Age", min_value=10, max_value=90, value=40, step=1)
type_of_contract = st.selectbox("Type Of Contract", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", ["1", "2", "3"])
duration_of_pitch= st.number_input("Duration Of Pitch", min_value=1, max_value=100, value=10, step=1)
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=10, value=1, step=1)
number_of_followups = st.number_input("Number Of Followups", min_value=1, max_value=10, value=1, step=1)
product_pitched = st.selectbox("Product Pitched", ["Deluxe", "Standard", "Basic"])
preferred_property_star = st.selectbox("Preferred Property Star", ["3", "4", "5"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
number_of_trips = st.number_input("Number Of Trips", min_value=1, max_value=10, value=1, step=1)
passport = st.selectbox("Passport", ["1", "0"])
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=10, value=1, step=1)
own_car = st.selectbox("Own Car", ["1", "0"])
number_of_children_visiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=10, value=0, step=1)
designation = st.selectbox("Designation", ["Executive", "Senior Manager", "Manager", "Senior Executive", "Junior Executive"])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=5000, step=100)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contract,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])


if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Package Buy" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")

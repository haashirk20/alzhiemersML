import streamlit as st
import numpy as np
from ML import Logistic_ML


def convert_to_array(gender, educationLevel, ethnicity):
    gender_array = []

    if gender == "Male":
        gender_array = [1, 0]
    else:
        gender_array = [0, 1]

    education_array = []
    if educationLevel == "No Degree":
        education_array = [1, 0, 0, 0]
    elif educationLevel == "High School":
        education_array = [0, 1, 0, 0]
    elif educationLevel == "Bachelor's":
        education_array = [0, 0, 1, 0]
    else:
        education_array = [0, 0, 0, 1]

    ethnicity_array = []
    print(ethnicity)
    if ethnicity == "White":
        ethnicity_array = [1, 0, 0, 0]
    if ethnicity == "Black":
        ethnicity_array = [0, 1, 0, 0]
    if ethnicity == "Asian":
        ethnicity_array = [0, 0, 1, 0]
    if ethnicity == "Other":
        ethnicity_array = [0, 0, 0, 1]

    return gender_array, education_array, ethnicity_array

# Streamlit app
st.set_page_config(layout="wide", page_title="Alzheimer's Disease Prediction", page_icon=":brain:")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.title("Alzheimer's Disease Prediction using Logistic Regression")


# ['Age', 'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity',
#        'DietQuality', 'SleepQuality', 'FamilyHistoryAlzheimers',
#        'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury',
#        'Hypertension', 'MemoryComplaints', 'BehavioralProblems', 'Confusion',
#        'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
#        'Forgetfulness', 'Gender_Female', 'Gender_Male',
#        'EducationLevel_College', 'EducationLevel_Graduate',
#        'EducationLevel_High School', 'EducationLevel_No Degree',
#        'Ethnicity_Asian', 'Ethnicity_Black', 'Ethnicity_Other',
#        'Ethnicity_White']

st.subheader("Please fill out the following information to get your prediction. The more accurate the information, the more accurate the prediction will be.")
st.subheader("Please note that this is not a diagnosis, but a prediction based on the information you provide.")

# columns
col1, col2 = st.columns(2)
butcol1, butcol2 = st.columns(2)



with col1:

    # Number input fields
    age = st.number_input("How old are you?", 0, 100, 0)
    BMI = st.number_input("What is your BMI?", 0, 100, 0)

    # select boxes
    gender = st.selectbox("What is your Gender?", ("Male", "Female"))
    educationLevel = st.selectbox("What is your highest level of Education?", ("No Degree", "High School", "Bachelor's", "Higher"))
    ethnicity = st.selectbox("What is your ethnicity?", ("White", "Black", "Asian", "Other"))

with col2:

    # sliders
    st.write("On a scale of 0 to 10, how would you rate the following?")
    alcoholConsumption = st.slider("Alcohol Consumption", 0, 10, 0)
    physicalActivity = st.slider("Physical Activity", 0, 10, 0)
    dietQuality = st.slider("Diet Quality", 0, 10, 0)
    sleepQuality = st.slider("Sleep Quality", 0, 10, 0)

with butcol1:
    # Radio buttons
    smoking = st.radio("Do you smoke?", ["Yes", "No"], horizontal = True)
    familyHistoryAlzheimers = st.radio("Do you have a family history of Alzheimers?", ["Yes", "No"], horizontal = True)
    cardiovascularDisease = st.radio("Do you have a Cardiovascular Disease?", ["Yes", "No"], horizontal = True)
    diabetes = st.radio("Do you have Diabetes?", ["Yes", "No"], horizontal = True)
    depression = st.radio("Have you been diagnosed with depression?", ["Yes", "No"], horizontal = True)
    headInjury = st.radio("Have you ever had a serious head injury?", ["Yes", "No"], horizontal = True)
    hypertension = st.radio("Do you have high blood pressure?", ["Yes", "No"], horizontal = True)
with butcol2:
    memoryComplaints = st.radio("Do you have poor memory?", ["Yes", "No"], horizontal = True)
    behavioralProblems = st.radio("Do you have a history of behavourial problems?", ["Yes", "No"], horizontal = True)
    confusion = st.radio("Do you find yourself confused throughout the day?", ["Yes", "No"], horizontal = True)
    disorientation = st.radio("Do you experience disorientation?", ["Yes", "No"], horizontal = True)
    personalityChanges = st.radio("Do you have frequent personality changes?", ["Yes", "No"], horizontal = True)
    difficultyCompletingTasks = st.radio("Would you say you have difficulty completing daily tasks?", ["Yes", "No"], horizontal = True)
    forgetfulness = st.radio("Would you consider yourself to be forgetful?", ["Yes", "No"], horizontal = True)

# convert radio buttons to 1 or 0
smoking = 1 if smoking == "Yes" else 0
familyHistoryAlzheimers = 1 if familyHistoryAlzheimers == "Yes" else 0
cardiovascularDisease = 1 if cardiovascularDisease == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
depression = 1 if depression == "Yes" else 0
headInjury = 1 if headInjury == "Yes" else 0
hypertension = 1 if hypertension == "Yes" else 0
memoryComplaints = 1 if memoryComplaints == "Yes" else 0
behavioralProblems = 1 if behavioralProblems == "Yes" else 0
confusion = 1 if confusion == "Yes" else 0
disorientation = 1 if disorientation == "Yes" else 0
personalityChanges = 1 if personalityChanges == "Yes" else 0
difficultyCompletingTasks = 1 if difficultyCompletingTasks == "Yes" else 0
forgetfulness = 1 if forgetfulness == "Yes" else 0

if st.button("Calculate"):
    ml = Logistic_ML()
    # add all values to np array and convert to float
    datapoint = np.array([age, BMI, smoking, alcoholConsumption, physicalActivity, dietQuality, sleepQuality, familyHistoryAlzheimers, cardiovascularDisease, diabetes, depression, headInjury, hypertension, memoryComplaints, behavioralProblems, confusion, disorientation, personalityChanges, difficultyCompletingTasks, forgetfulness])
    gender_array, education_array, ethnicity_array = convert_to_array(gender, educationLevel, ethnicity)
    gender_array = np.array(gender_array)
    education_array = np.array(education_array)
    ethnicity_array = np.array(ethnicity_array)
    #print(gender_array, education_array, ethnicity_array)
    datapoint = np.concat((datapoint, gender_array, education_array, ethnicity_array))
    #print(datapoint)
    result = ml.predict(datapoint.reshape(1, -1))
    print(result[0])
    diagnosis = round(result[0][1] * 100,2)
    st.success(f"You have a {diagnosis}% chance of getting Alzheimer's at some point throughout your life.")

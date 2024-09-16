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
st.title("Unnamed Function Calculator")


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

# Number input fields
age = st.number_input("Age", 0, 100, 0)
BMI = st.number_input("BMI", 0, 100, 0)

# select boxes
gender = st.selectbox("Gender", ("Male", "Female"))
educationLevel = st.selectbox("Education Level", ("No Degree", "High School", "Bachelor's", "Higher"))
ethnicity = st.selectbox("ethnicity", ("White", "Black", "Asian", "Other"))

# sliders
alcoholConsumption = st.slider("Alcohol Consumption", 0, 10, 0)
physicalActivity = st.slider("Physical Activity", 0, 10, 0)
dietQuality = st.slider("Diet Quality", 0, 10, 0)
sleepQuality = st.slider("Sleep Quality", 0, 10, 0)
# Radio buttons
smoking = st.radio("Smoking", ["Yes", "No"])
familyHistoryAlzheimers = st.radio("Family History Alzheimers", ["Yes", "No"])
cardiovascularDisease = st.radio("Cardiovascular Disease", ["Yes", "No"])
diabetes = st.radio("Diabetes", ["Yes", "No"])
depression = st.radio("Depression", ["Yes", "No"])
headInjury = st.radio("Head Injury", ["Yes", "No"])
hypertension = st.radio("Hypertension", ["Yes", "No"])
memoryComplaints = st.radio("Memory Complaints", ["Yes", "No"])
behavioralProblems = st.radio("Behavioral Problems", ["Yes", "No"])
confusion = st.radio("Confusion", ["Yes", "No"])
disorientation = st.radio("Disorientation", ["Yes", "No"])
personalityChanges = st.radio("Personality Changes", ["Yes", "No"])
difficultyCompletingTasks = st.radio("Difficulty Completing Tasks", ["Yes", "No"])
forgetfulness = st.radio("Forgetfulness", ["Yes", "No"])

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
    #ml = Logistic_ML()
    # add all values to np array and convert to float
    datapoint = np.array([age, BMI, smoking, alcoholConsumption, physicalActivity, dietQuality, sleepQuality, familyHistoryAlzheimers, cardiovascularDisease, diabetes, depression, headInjury, hypertension, memoryComplaints, behavioralProblems, confusion, disorientation, personalityChanges, difficultyCompletingTasks, forgetfulness])
    gender_array, education_array, ethnicity_array = convert_to_array(gender, educationLevel, ethnicity)
    gender_array = np.array(gender_array)
    education_array = np.array(education_array)
    ethnicity_array = np.array(ethnicity_array)
    print(gender_array, education_array, ethnicity_array)
    datapoint = np.concat((datapoint, gender_array, education_array, ethnicity_array))
    print(datapoint)
    #result = ml.predict(datapoint)
    #st.success(f"Result: {result}")

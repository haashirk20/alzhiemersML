import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

#load alzhimers data
data = pd.read_csv('alzheimers_disease_data.csv')

#split data into training, and testing
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


def get_input_targets(data):
    # extract the target vector
    t = np.array(data["Diagnosis"])

    # extract the data matrix:
    X_feats = np.array(data[["Age","BMI","Smoking","AlcoholConsumption","PhysicalActivity","DietQuality","SleepQuality","FamilyHistoryAlzheimers","CardiovascularDisease","Diabetes","Depression","HeadInjury","Hypertension","SystolicBP","DiastolicBP","CholesterolTotal","CholesterolLDL","CholesterolHDL","CholesterolTriglycerides","MMSE","FunctionalAssessment","MemoryComplaints","BehavioralProblems","ADL","Confusion","Disorientation","PersonalityChanges","DifficultyCompletingTasks","Forgetfulness"]])

    n = len(data) # number of data points, you may find this information useful
    print(n)
    #X = np.concatenate((np.ones((n,1)), X_feats), axis=1)
    X = X_feats
    return (X, t)

X_train, t_train = get_input_targets(train_data)
#X_valid, t_valid = get_input_targets(validation_data)
X_test, t_test = get_input_targets(test_data)

log_model = LogisticRegression(random_state=16)

log_model.fit(X_train, t_train)

# hyperparameter tuning


y_pred = log_model.predict(X_test)

from sklearn.metrics import classification_report
target_names = ['without alzheimers', 'with alzheimers']
print(classification_report(t_test, y_pred, target_names=target_names))

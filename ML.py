import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

def get_input_targets(data):
    # extract the target vector
    t = np.array(data["Diagnosis"])

# "Age","BMI","Smoking","AlcoholConsumption",
#                              "PhysicalActivity","DietQuality","SleepQuality",
#                              "FamilyHistoryAlzheimers","CardiovascularDisease",
#                              "Diabetes","Depression","HeadInjury","Hypertension",
#                              "SystolicBP","DiastolicBP","CholesterolTotal",
#                              "CholesterolLDL","CholesterolHDL","CholesterolTriglycerides",
#                              "MMSE","FunctionalAssessment","MemoryComplaints",
#                              "BehavioralProblems","ADL","Confusion",
#                              "Disorientation","PersonalityChanges",
#                              "DifficultyCompletingTasks","Forgetfulness"

    X_feats = np.array(data[["Age","BMI","Smoking","AlcoholConsumption",
                                "PhysicalActivity","DietQuality","SleepQuality",
                                "FamilyHistoryAlzheimers","CardiovascularDisease",
                                "Diabetes","Depression","HeadInjury","Hypertension","MemoryComplaints",
                                "BehavioralProblems","ADL","Confusion",
                                "Disorientation","PersonalityChanges",
                                "DifficultyCompletingTasks","Forgetfulness"]])

    # # extract the data matrix:
    X_feats = np.array(data[["Age","BMI","Smoking","AlcoholConsumption",
                                 "PhysicalActivity","DietQuality","SleepQuality",
                                 "FamilyHistoryAlzheimers","CardiovascularDisease","Hypertension"]])

    # normalize the data
    # X_feats = (X_feats - np.mean(X_feats, axis=0)) / np.std(X_feats, axis=0)
    n = len(data) # number of data points, you may find this information useful
    print(n)
    #X = np.concatenate((np.ones((n,1)), X_feats), axis=1)
    X = X_feats
    return (X, t)

def train_model(data):
    X_train, t_train = get_input_targets(data)
    log_model = LogisticRegression(random_state=16)
    log_model.fit(X_train, t_train)
    return log_model

def predict(X):
    data = pd.read_csv('alzheimers_disease_data.csv')
    log_model = train_model(data)
    return log_model.predict_proba(X)

def log_regression_metrics(data):
    X_train, t_train = get_input_targets(data)
    log_model = LogisticRegression(random_state=16, solver='saga', penalty='elasticnet', l1_ratio=0.5)
    log_model.fit(X_train, t_train)
    X_test, t_test = get_input_targets(data)
    y_pred = log_model.predict(X_test)
    mse = mean_squared_error(t_test, y_pred)
    r2 = r2_score(t_test, y_pred)

    from sklearn.metrics import classification_report
    target_names = ['without alzheimers', 'with alzheimers']
    print(classification_report(t_test, y_pred, target_names=target_names))

    return (mse, r2)



class Logistic_ML:
    def __init__(self):
        data = pd.read_csv('alzheimers_disease_data.csv')
        self.log_model = train_model(data)
    
    def predict(self, X):
        return self.log_model.predict_proba(X)
    
if __name__ == '__main__':
    X = np.array([[51, 28,  1, 10,  5,  5,  7,  1,  1,  1]])
    #print(X)
    ml = Logistic_ML()
    print(ml.predict(X))
    data = pd.read_csv('alzheimers_disease_data.csv')
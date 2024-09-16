import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

def get_input_targets(data):

    # extract the target vector
    t = np.array(data["Diagnosis"])

    #data = clean_data(data)
    # grab all columns and put into array
    data = data.drop(columns=["Diagnosis"])
    X_feats = np.array(data)

    # normalize the data
    # X_feats = (X_feats - np.mean(X_feats, axis=0)) / np.std(X_feats, axis=0)
    n = len(data) # number of data points, you may find this information useful
    X = X_feats
    return (X, t)

def clean_data(data):
    # in gender row convert 0 to male and 1 to female
    data['Gender'] = data['Gender'].replace({0: 'Male', 1: 'Female'})
    # in educationlevel row convert 0 to no degree, 1 to high school, 2 to college, 3 to graduate
    data['EducationLevel'] = data['EducationLevel'].replace({0: 'No Degree', 1: 'High School', 2: 'College', 3: 'Graduate'})
    # change ethnicity from 0 to white, 1 to black, 2 to asian, 3 to other
    data['Ethnicity'] = data['Ethnicity'].replace({0: 'White', 1: 'Black', 2: 'Asian', 3: 'Other'})
    # use one hot encoding for gender, educationlevel and ethnicity
    data = pd.get_dummies(data, columns=["Gender", "EducationLevel", "Ethnicity"])
    # convert all trues to 1 and false to 0
    data = data * 1
    data.drop(columns=["SystolicBP","DiastolicBP","CholesterolTotal",
                                "CholesterolLDL","CholesterolHDL","CholesterolTriglycerides",
                                "MMSE","FunctionalAssessment", "ADL", "DoctorInCharge", "PatientID"], inplace=True)
    print(data.columns)
    return data

def train_model(data):
    X_feats, t = get_input_targets(data)
    X_train, X_test, t_train, t_test = train_test_split(X_feats, t, test_size=0.2, random_state=16)
    log_model = LogisticRegression(random_state=16, solver='saga', penalty='elasticnet', l1_ratio=0.5, max_iter=3000)
    log_model.fit(X_train, t_train)
    return log_model

def tune_model(data):
    model = LogisticRegression(max_iter=10000)
    parameters = {'solver': ['saga'], "C":np.logspace(-3,3,7), "penalty":["l1","l2", "elasticnet"], "l1_ratio":np.linspace(0,1,10)}
    
    data = clean_data(data)
    X_feats, t = get_input_targets(data)
    clf = GridSearchCV(estimator=model, param_grid=parameters, cv=3, verbose=True, n_jobs=-1, scoring='f1_micro')
    best_clf = clf.fit(X_feats, t)
    print(best_clf.best_estimator_)
    print(f'Best score: {best_clf.best_score_}')

def log_regression_metrics(data):
    data = clean_data(data)
    X_feats, t = get_input_targets(data)
    X_train, X_test, t_train, t_test = train_test_split(X_feats, t, test_size=0.2, random_state=16)
    log_model = LogisticRegression()
    log_model = log_model.fit(X_train, t_train)
    y_pred = log_model.predict(X_test)
    mse = mean_squared_error(t_test, y_pred)
    r2 = r2_score(t_test, y_pred)

    from sklearn.metrics import classification_report
    target_names = ['without alzheimers', 'with alzheimers']
    print(classification_report(t_test, y_pred, target_names=target_names))

    return (mse, r2)
class Logistic_ML:
    def __init__(self):
        data = clean_data(pd.read_csv('alzheimers_disease_data.csv'))
        self.log_model = train_model(data)
    
    def predict(self, X):
        return self.log_model.predict_proba(X)
    
    
if __name__ == '__main__':
    X = np.array([[60, 30, 1, 5, 2, 5, 4, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]])
    #print(X)
    #ml = Logistic_ML()
    #print(ml.predict(X))
    data = pd.read_csv('alzheimers_disease_data.csv')
    tune_model(data)
    #print(log_regression_metrics(data)) 
    #clean_data(data)
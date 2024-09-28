from flask import Flask, render_template, request
from ML import Logistic_ML
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    print("hi")
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Get input data from the form
    print("hi2")
    var1 = int(request.form['age'])
    var2 = int(request.form['bmi'])
    var3 = int(request.form['smoking'])
    # ["Age","BMI","Smoking","AlcoholConsumption",
    #                              "PhysicalActivity","DietQuality","SleepQuality",
    #                              "FamilyHistoryAlzheimers","CardiovascularDisease","Hypertension"]
    var4 = int(request.form['alcoholConsumption'])
    var5 = int(request.form['physicalActivity'])
    var6 = int(request.form['dietQuality'])
    var7 = int(request.form['sleepQuality'])
    var8 = int(request.form['familyHistoryAlzheimers'])
    var9 = int(request.form['cardiovascularDisease'])
    var10 = int(request.form['hypertension'])
    # ... Get other variables as needed ...

    # Prepare input data for your ML model
    input_data = np.array([[var1, var2, var3, var4, var5, var6, var7, var8, var9, var10]])  # Adjust based on your model's requirements

    # Make prediction using your ML model
    ml = Logistic_ML()

    prediction = ml.predict(input_data)
    print(prediction)

    # Render the result template
    return render_template('predict.html', prediction=prediction)
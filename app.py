from flask import Flask,request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd 
import math
import os
from src.exception import CustomException
from src.logger import logging
from src.pipelines.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application
CORS(app)



# Route for a home page
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Loan Eligibility Prediction API!"})


# Function to categorize 'age'
def categorize_age(age):
    if age < 26:
        return "Young"
    elif age < 51:
        return "Middle"
    else:
        return "Senior"

# Function to categorize 'bmi'
def categorize_bmi(bmi):
    if bmi < 18.6:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obesity"

# Route for making predictions
@app.route('/predictdata', methods=['POST'])
def predictdata():
    try:
        #log the route hit
        print("hitting the route successfully...")  
        # Retrieve data from the request JSON
        data = request.json
        
        
        # Convert age and bmi to categories
        age_category = categorize_age(int(data.get('age')))
        bmi_category = categorize_bmi(float(data.get('bmi')))

        # Create a CustomData instance with the input data
        custom_data = CustomData(
            gender=data.get('gender'),
            age=age_category,
            smoking=data.get('smoking'),
            hypertension=int(data.get('hypertension')),
            heart_disease=int(data.get('heart_disease')),
            bmi=bmi_category,
            hemoglobin_level=float(data.get('hemoglobin_level')),
            blood_glucose_level=float(data.get('blood_glucose_level'))
        )


        
        # Convert the input data to a DataFrame
        pred_df = custom_data.make_data_frame()
        
        # Initialize the prediction pipeline and make a prediction
        predict_pipeline = PredictPipeline()
        # Make predictions
        predictions = predict_pipeline.predict_fraud(pred_df)

        print(f"result is {predictions[0]}")

        if(predictions[0] ==0):
            return jsonify({"result":"Doesn't have Diabetes"})
        else:
            return jsonify({"result":"Has Diabetes"})
        

    except Exception as e:
        raise CustomException(e)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)





    # body should like this 

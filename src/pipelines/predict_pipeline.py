import pandas as pd
import math
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict_diabetes(self, features):
        try:
            model_path = 'model/Gradient_Boosting_model.pkl'
            preprocessor_path = 'model/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Loading model and preprocessor file...")

            # Transform the input data
            data_scaled = preprocessor.transform(features)
            data_scaled_df = pd.DataFrame(data_scaled, columns=preprocessor.feature_names_in_)
            preds = model.predict(data_scaled_df)

            logging.info("Prediction Completed...")

            return preds

        except Exception as e:
            raise CustomException(e)


class CustomData:
    def __init__(self,
                 gender: str,                     # 'male' / 'female' / 'other'
                 age: str,                        # Numeric value
                 smoking: str,                    # 'non_smoker', 'current_smoker', 'former_smoker', 'unknown'
                 hypertension: int,               # 0 or 1
                 heart_disease: int,              # 0 or 1
                 bmi: str,                      # Numeric value
                 hemoglobin_level: float,         # Numeric value
                 blood_glucose_level: float):     # Numeric value

        # Initialize attributes
        self.gender = gender
        self.age = age
        self.smoking = smoking
        self.hypertension = hypertension
        self.heart_disease = heart_disease
        self.bmi = bmi
        self.hemoglobin_level = hemoglobin_level
        self.blood_glucose_level = blood_glucose_level

    def make_data_frame(self):
        try:
            # Create a dictionary with the data
            custom_data_input_dict = {
                "gender": [self.gender],
                "age": [self.age],
                "smoking": [self.smoking],
                "hypertension": [self.hypertension],
                "heart_disease": [self.heart_disease],
                "bmi": [self.bmi],
                "hemoglobin_level": [self.hemoglobin_level],
                "blood_glucose_level": [self.blood_glucose_level]
            }

            # Return a DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e)


# Main function for testing
if __name__ == "__main__":
    try:
        # Create a CustomData instance with sample data
        custom_data = CustomData(
            gender="Female",
            age='Middle',
            smoking="current_smoker",
            hypertension=1,
            heart_disease=0,
            bmi='normal',
            hemoglobin_level=5.2,
            blood_glucose_level=150.0
        )

        # Convert the custom data to a DataFrame
        pred_df = custom_data.make_data_frame()

        # Initialize the PredictPipeline
        predict_pipeline = PredictPipeline()

        # Make predictions
        predictions = predict_pipeline.predict_diabetes(pred_df)

        if predictions[0] == 0:
            print("Doesn't have Diabetes")
        else:
            print("Has Diabetes")
    except Exception as e:
        print(f"An error occurred: {e}")

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, FunctionTransformer
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_file_path = 'artifacts\preprocessor.pkl'

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    

    def data_transformer_pipeline(self):
        try:
            # Define categorical columns
            cat_cols = ['gender', 'age', 'smoking', 'hypertension', 'heart_disease', 'bmi']

            # Define numerical columns
            num_cols = ['hemoglobin_level', 'blood_glucose_level']


            # Numerical pipeline (filling missing values, replacing -999/? to 0, scaling)
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Replace missing values with mean
                ('scaler', MinMaxScaler())  # Scale values between 0 and 1
            ])

            # Categorical pipeline (filling missing values, encoding)
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing with mode
                ('ordinal_encoder', OrdinalEncoder())  # Label encoding # Handle unknown categories
            ])

            # Combine pipelines for preprocessing
            preprocessor = ColumnTransformer(transformers=[
                    ('cat_pipeline', cat_pipeline, cat_cols),
                    ('num_pipeline', num_pipeline, num_cols),
                ], remainder = 'passthrough'
            )

            # Build complete pipeline
            pre_processing_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor)  # Apply preprocessing steps
            ])
            logging.info("Data transformation has been completed...")

            return pre_processing_pipeline

        except Exception as e:
            raise CustomException(e)


    def add_new_features(self, df):
        """
        Add new features to the dataset, including the 'approved' column and transforming 'age'.
        """
        try:

            df = df.drop_duplicates()

            df['age'] = pd.to_numeric(df['age'], errors='coerce')


            # Transform 'age' into categories
            def categorize_age(age):
                if age < 26:
                    return "Young";
                elif age < 51: 
                    return 'Middle';
                else: return 'Senior'


            def categorize_bmi(value):
                if value < 18.6:
                    return "underweight";
                elif value < 25: 
                    return 'normal';
                elif value < 30: 
                    return 'overweight';
                else: return 'obesity'


            def categorize_smoking(smoking_status):
                if smoking_status in ['never', 'not current']:
                    return 'non_smoker'
                elif smoking_status == 'current':
                    return 'current_smoker'
                elif smoking_status in ['former', 'ever']:
                    return 'former_smoker'
                else:
                    return 'unknown'  # Handle 'No Info'


            df['age'] = df['age'].apply(categorize_age)
            df['bmi'] = df['bmi'].apply(categorize_bmi)
            df['smoking'] = df['smoking'].apply(categorize_smoking)

            print(df.shape)
            

            

            # print("adding feature complete sucessfully")
            return df
        except Exception as e:
            raise CustomException(e)

        
    def initiate_data_transformation(self, raw_data_path):
        try:
            logging.info("Loading dataset...")
            data = pd.read_csv(raw_data_path)

            # Debugging: Check raw data
            # print("Raw Data Head:\n", data.head())

            # Define target columns
            target_column = "diabetes"
            

            logging.info("Adding new features...")
            data = self.add_new_features(data)

            # Debugging: Check data after adding features
            # print(data.shape)

            # Create preprocessing pipeline
            preprocessor = self.data_transformer_pipeline()

            # Split features and targets
            X = data.drop(columns=[target_column], axis=1)
            y = data[target_column]

            # print(X.head(5))

            X_transformed = preprocessor.fit_transform(X)
            # print(X_transformed)


            # Combine the transformed data into a DataFrame
            feature_columns = X.columns
            X_transformed_df = pd.DataFrame(X_transformed, columns=feature_columns)

            # Rearrange columns to match the original order
            # X_transformed_df = X_transformed_df[X.columns]  # Ensure same order as original


            # Add targets back to the processed DataFrame
            X_transformed_df['diabetes'] = y.values

            print(X_transformed_df.head(5))
            # print(X_transformed_df.shape)

            logging.info("Saving preprocessor object...")
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor
            )

            return X_transformed_df, self.data_transformation_config.preprocessor_file_path
        except Exception as e:
            raise CustomException(e)



if __name__ =='__main__':
    raw_data_path = 'notebook\data\diabetes-prediction-dataset.csv'
    try:
        # Initialize the DataTransformation class
        data_transformation = DataTransformation()

        # Call the initiate_data_transformation method and get the results
        processed_data, preprocessor_file_path = data_transformation.initiate_data_transformation(raw_data_path)

        # Print the processed data and the path to the saved pipeline
        # print("\nProcessed Data Head:")
        # print(processed_data.head(4))  # Print the first few rows of the processed data

        # print("\nSaved Preprocessor Path:")
        # print(preprocessor_file_path)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
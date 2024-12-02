from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

class ModelConfig:
    trained_model_path:str = 'artifacts\model.pkl'

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelConfig()

    def initiate_model_trainer(self, data):
        try:
            target_column = 'diabetes'
            X = data.drop(columns = [target_column], axis = 1)
            y = data[target_column]
            

            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

            # Define models and parameters
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
            }


            model_report:dict = evaluate_models(X_train, X_test, y_train, y_test, models)

            # Find the best model based on the test score (second element in the tuple)
            best_model_name, (best_train_score, best_test_score) = max(
                model_report.items(), key=lambda x: x[1][1]  # Use test score for max
            )

            best_model = models[best_model_name]

            # Assign best_test_score as best_model_score for comparison
            best_model_score = best_test_score

            # Check if the best test score is above the threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            # # Save the best model
            # save_object(
            #     file_path=self.model_trainer_config.trained_model_path,
            #     obj=best_model
            # )

            # Return the report, best model name, and R2 score for the best model
            return model_report, best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e)
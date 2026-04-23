import os
import sys
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass(frozen=True)
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(random_state=42, verbosity=0),
                "CatBoost Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            params = {
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10],
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5, 10, 15],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [50, 100, 200],
                    "subsample": [0.8, 1.0],
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
                "XGBoost Regressor": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7],
                },
                "CatBoost Regressor": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [50, 100, 200],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.01, 0.1, 1.0],
                    "n_estimators": [50, 100, 200],
                },
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            print("\nModel Report:")
            for model_name, score in model_report.items():
                print(f"{model_name}: {score:.4f}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            print(f"\nBest Model: {best_model_name}")
            print(f"Best R2 Score: {best_model_score:.4f}")

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable score", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return best_model_name, r2_square

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.Components.data_ingestion import DataIngestion
    from src.Components.data_transformation import DataTransformation

    ingestion_obj = DataIngestion()
    train_path, test_path, _ = ingestion_obj.initiate_data_ingestion()

    transformation_obj = DataTransformation()
    train_arr, test_arr, _ = transformation_obj.initiate_data_transformation(train_path, test_path)

    model_trainer_obj = ModelTrainer()
    best_model_name, r2_score_value = model_trainer_obj.initiate_model_trainer(train_arr, test_arr)

    print("\nFinal Result")
    print("Best Model Name:", best_model_name)
    print("Final R2 Score:", round(r2_score_value, 4))
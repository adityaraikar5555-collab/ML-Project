import os
import sys
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        logging.info(f"Saving object at: {file_path}")

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Object saved successfully")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        logging.info(f"Loading object from: {file_path}")

        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Trains models, performs GridSearchCV where params exist,
    and returns a dictionary of model name and test R2 score.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")

            param_grid = params.get(model_name, {})

            if param_grid:
                gs = GridSearchCV(
                    model,
                    param_grid,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1,
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                models[model_name] = best_model
            else:
                best_model = model
                best_model.fit(X_train, y_train)
                models[model_name] = best_model

            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            logging.info(f"{model_name} R2 score: {test_model_score}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
import os
import sys
from dataclasses import dataclass
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "regression_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path=None):
        try:
            logging.info("Split train and test input data")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "XGBRegressor": XGBRegressor()
            }

            logging.info("Training and evaluating models")
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models=models)

            best_model_name, best_score = max(
                ((name, scores["test_score"]) for name, scores in model_report.items()),
                key=lambda item: item[1]
            ) 

            if best_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model: {best_model_name}, Best Scores: {model_report[best_model_name]}")

            best_model = models[best_model_name]
            save_obj(obj=best_model, file_path=self.model_trainer_config.trained_model_path)
            logging.info("Saved best model as a pickle file")

            best_model_pred = best_model.predict(X_test)
            best_model_r2_score = r2_score(y_test, best_model_pred)

        except Exception as e:
            raise CustomException(e, sys)
        
        return best_model_r2_score

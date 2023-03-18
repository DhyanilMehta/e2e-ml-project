import os
import sys
import dill
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exceptions import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


def load_obj(file_path: str):
    try:
        with open(file_path, "rb") as f:
            obj = dill.load(f)
    
    except Exception as e:
        raise CustomException(e, sys)
    
    return obj

def save_obj(obj: object, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params=None):
    try:
        report = {}

        for model_name, model in tqdm(models.items(), desc="Training and evaluating"):
            
            if params:
                logger.info("Performing hyper-parameter tuning using grid search")
                
                param = params[model_name]
                grid_search = GridSearchCV(model, param)
                grid_search.fit(X_train, y_train)

                model.set_params(**grid_search.best_params_)
            
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = {"train_score": train_score, "test_score": test_score}

    except Exception as e:
        raise CustomException(e, sys)

    return report

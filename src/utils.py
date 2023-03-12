import os
import sys
import dill
import numpy as np
import pandas as pd

from src.exceptions import CustomException

def save_obj(obj: object, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    
    except Exception as e:
        raise CustomException(e, sys)

import os
import sys
import pandas as pd

from src.exceptions import CustomException
from src.utils import load_obj


class PredictPipeline:
    def __init__(self):
        self.preprocessor = load_obj(os.path.join("artifacts", "preprocessor.pkl"))
        self.model = load_obj(os.path.join("artifacts", "regression_model.pkl"))

    def predict(self, features):
        try:
            scaled_features = self.preprocessor.transform(features)
            prediction = self.model.predict(scaled_features)
        except Exception as e:
            raise CustomException(e, sys)
        
        return prediction

class CustomDataParser:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_dataframe(self):
        try:
            df = pd.DataFrame({
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            })
        except Exception as e:
            raise CustomException(e, sys)
        
        return df

import os
import sys
import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import get_logger
from src.exceptions import CustomException
from src.utils import save_obj

logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    data_transformer_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer(self, numerical_features: List[str], categorical_features: List[str]):
        """
        This function is responsible for generating a data transformation pipeline for numerical and categorical features.

        Args:
            numerical_features (list[str]): A list of numerical features to transform
            categorical_features (list[str]): A list of categorical features to transform
        
        Raises:
            CustomException: Raise custom exception in-case something fails

        Returns:
            ColumnTransformer: Returns a ColumnTransformer with preprocessing pipelines for numerical and categorical features
        """
        try:
            logger.info(f"Numerical features: {numerical_features}")
            logger.info(f"Categorical features: {categorical_features}")

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            logger.info("Numerical features pipeline created")

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logger.info("Categorical features pipeline created")

            data_transformer = ColumnTransformer([
                ("numerical_pipeline", numerical_pipeline, numerical_features),
                ("categorical_pipeline", categorical_pipeline, categorical_features)
            ])
            logger.info("Data transformer created by combining pipelines")
        
        except Exception as e:
            raise CustomException(e, sys)

        return data_transformer
    
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logger.info("Reading train and test data completed")

            logger.info("Getting data transformer")
            target_column = "math score"
            numer_features = ["writing score", "reading score"]
            cat_features = [
                "lunch", "test preparation course",
                "gender", "race/ethnicity", "parental level of education",
            ]

            data_transformer = self.get_data_transformer(numer_features, cat_features)

            input_feature_train_df = train_df.drop(target_column, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(target_column, axis=1)
            target_feature_test_df = test_df[target_column]

            logger.info("Applying data transformer on train and test dataframes")
            input_feature_train_arr = data_transformer.fit_transform(input_feature_train_df)
            input_feature_test_arr = data_transformer.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(obj=data_transformer, file_path=self.transformation_config.data_transformer_path)
            logger.info("Saved data transformer as a pickle file")

        except Exception as e:
            raise CustomException(e, sys)
        
        return train_arr, test_arr, self.transformation_config.data_transformer_path


if __name__ == "__main__":
    pass

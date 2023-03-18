import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.exceptions import CustomException
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Initiating data ingestion")
        try:
            df = pd.read_csv("notebooks/data/students_performance.csv")
            logger.info("Imported the dataset as dataframe")

            os.makedirs(os.path.dirname(
                self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)
            logger.info("Saved the dataset as artifact")

            logger.info("Splitting dataset into train-test")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)
            logger.info("Data ingestion completed")
        except Exception as e:
            raise CustomException(e, sys)

        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path,
            self.ingestion_config.raw_data_path
        )


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path, test_path, raw_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_path, test_path)

    model_trainer = ModelTrainer()
    print("R2 Score:", model_trainer.initiate_model_trainer(train_arr, test_arr))

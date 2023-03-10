import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exceptions import CustomException


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            df = pd.read_csv("notebooks/data/students_performance.csv")
            logging.info("Imported the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved the dataset as artifact")

            logging.info("Splitting dataset into train-test")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed")
        except Exception as e:
            raise CustomException(e, sys)

        return (
            self.ingestion_config.train_data_path, 
            self.ingestion_config.test_data_path, 
            self.ingestion_config.raw_data_path
        )


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()

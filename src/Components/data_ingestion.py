import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass(frozen=True)
class DataIngestionConfig:
    artifacts_dir: str = "artifacts"
    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")
    raw_data_path: str = os.path.join(artifacts_dir, "raw.csv")
    source_data_path: str = os.path.join("Notebook", "stud.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")

        try:
            os.makedirs(self.config.artifacts_dir, exist_ok=True)
            logging.info("Artifacts directory created successfully")

            df = pd.read_csv(self.config.source_data_path)
            logging.info("Dataset read as dataframe successfully")
            logging.info(f"Dataset shape: rows={df.shape[0]}, columns={df.shape[1]}")

            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info("Raw dataset saved successfully")

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info("Train and test datasets saved successfully")
            logging.info("Data ingestion completed")

            return (
                self.config.train_data_path,
                self.config.test_data_path,
                self.config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    print(obj.initiate_data_ingestion())
import os
import sys
from src.logger.logging import logging
from src.exception.exception import customException

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts", "raw.csv")
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            data = pd.read_csv("data/raw.csv")      # load raw data
            logging.info("reading data")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)      # make the artifact folder
            data.to_csv(self.ingestion_config.raw_data_path, index=False)       # save the raw data to artifact folder
            logging.info("saved the raw dataset in artifact folder")
            
            logging.info("starting train-test split")
            
            train_data, test_data = train_test_split(data, test_size=0.25)      # train-test split on the raw data
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)   # save train data to artifact folder
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)     # save test data to artifact folder
            
            logging.info("Data Ingestion completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info()
            raise customException(e, sys)




if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
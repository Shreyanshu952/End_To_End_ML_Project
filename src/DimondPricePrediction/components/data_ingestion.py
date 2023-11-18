'''
Module is nothing but a python file (.py file). For example - In this gemstone price prediction project we have different different modules like data ingestion module, data transformation module, model trainer module etc.

Modular coding means we are going to segregate our tasks in different different modules (i.e. .py files).

Artifacts is the folder containing (or collecting) the physical outputs of components (i.e. data_ingestion, data_transformation and model_trainer).
It is a concept used in the world of machine learning.
'''



import pandas as pd
import numpy as np

from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from sklearn.model_selection import train_test_split

import os
import sys
from pathlib import Path

# creating configuration class for saving datasets into artifacts folder
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts", "raw.csv")
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")


# creating a class for loading dataset
class DataIngestion:
    def __init__(self):
        #creating an instance for DataIngestionConfig class
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("data ingestion started")

        try:
            #reading the dataset
            data = pd.read_csv(Path(os.path.join("notebooks/data", "gemstone_train.csv")))
            logging.info("i have read the dataset as a DataFrame")

            # making artifacts folder and directory
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("I have saved the raw data into artifacts folder")

            logging.info("Now i am performing train test split of gemstone dataset")
            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
            logging.info("train test split of data is completed")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            logging.info("I have saved the train data into artifacts folder")
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("I have saved the test data into artifacts folder")

            logging.info("data ingestion part is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occured at data ingestion stage")
            raise customexception(e, sys)
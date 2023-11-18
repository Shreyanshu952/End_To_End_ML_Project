import os
import sys

import pandas as pd

from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from src.DimondPricePrediction.components.data_ingestion import DataIngestion
from src.DimondPricePrediction.components.data_transformation import DataTransformation
from src.DimondPricePrediction.components.model_trainer import ModelTrainer

# creating an object of DataIngestion  and executing it
obj = DataIngestion()

train_data_path, test_data_path = obj.initiate_data_ingestion()

# creating an object of DataTransformation and executing it
data_transformation=DataTransformation()

train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)

# creating an object of ModelTrainer and executing it
model_trainer_obj=ModelTrainer()

model_trainer_obj.initiate_model_training(train_arr,test_arr)
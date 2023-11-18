import pandas as pd
import numpy as np

import os
import sys

from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from dataclasses import dataclass

from src.DimondPricePrediction.utils.utils import save_object
from src.DimondPricePrediction.utils.utils import evaluate_model

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

#creating the configuration of model.pkl file
@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join("artifacts", "model.pkl")

# creating a class for training the model
class ModelTrainer:
    def __init__(self):
        #initiating the object for ModelTrainerConfig
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            #separating independent and dependent features from train and test dataframe
            logging.info('Splitting Dependent and Independent variables from train and test data')

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            #defining a dictionary containing all the models
            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
            }

            #training and evaluating the models and doing prediction using the model
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'All Models Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            
            #saving the model.pkl file into artifacts folder
            save_object(
                 file_path=self.model_trainer_config.model_trainer_file_path,
                 obj=best_model
            )

            logging.info("model pickle file has been saved")

        except Exception as e:
            logging.info("Exception occured at model training stage")
            raise customexception(e, sys)
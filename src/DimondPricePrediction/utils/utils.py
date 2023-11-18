import os
import sys

import pickle

import numpy as np
import pandas as pd

from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Exception occured at saving a file object")
        raise customexception(e, sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        training_report = {}

        #creating loop to get r2_score of every model
        for i in range(len(models)):
            model = list(models.values())[i]

            #training the model
            model.fit(X_train, y_train)

            #getting the training score of the model
            train_model_score = model.score(X_train, y_train)

            #prediction using trained model
            y_test_pred = model.predict(X_test)

            # getting the r2_score for the test data
            test_model_score = r2_score(y_test, y_test_pred)

            #generating the report
            report[list(models.keys())[i]] =  test_model_score
            training_report[list(models.keys())[i]] =  train_model_score

        return report
    
    except Exception as e:
        logging.info("Exception occured at training, evaluating and prection stage of the model")
        raise customexception(e, sys)


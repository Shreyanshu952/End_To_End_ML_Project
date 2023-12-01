import os
import sys

import numpy as np

from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from src.DimondPricePrediction.utils.utils import load_object

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

from urllib.parse import urlparse



# creating a class for model evaluation
class ModelEvaluation:
    def __init__(self):
        pass
    
    #creating a definition of metric evaluation
    def evaluate_metric(self, actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        rsquared_value = r2_score(actual, predicted)
        return mae, rmse, rsquared_value

    #creating a definition for initializing model evaluation
    def initiate_model_evaluation(self, test_array):
        try:
            logging.info("Model Evaluation has been initiated")

            logging.info("Splitting test array into independent and dependent features")
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            logging.info("Splitting of test array has been done")

            #loading the best model from the artifacts folder
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)
            logging.info("Best Model has been loaded")

            #initiating mlflow run for tracking metrices
            with mlflow.start_run():
                #predicting the X_test values using our best model
                predicted_values = model.predict(X_test)

                #calculating metrices
                mae, rmse, rsquared_value = self.evaluate_metric(y_test, predicted_values)

                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2_score", rsquared_value)

        except Exception as e:
            logging.info("Error occured at model evaluation stage")
            raise customexception(e, sys)
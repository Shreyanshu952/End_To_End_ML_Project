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

            # mlflow keeps the track of experiments and also do model registry
            # if we use directly mlflow without the use of DAGshub then our code will be tracked only in local enviroment 
            # and we get ui which we can run only through local server and and we can not track our experiments using outside servers
            # for doing that, we have to launch our experiments on DAGshub
            # DAGshub can host our experiments and also register our model which can we seen to outside world
            # DAGshub is an open source platform

            #writing code for setting local environment to DAGshub
            mlflow.set_registry_uri("https://dagshub.com/Shreyanshu952/End_To_End_ML_Project.mlflow")
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            logging.info(f"Type of store is {tracking_url_type_store}")

            #initiating mlflow run for tracking metrices
            with mlflow.start_run():
                #predicting the X_test values using our best model
                predicted_values = model.predict(X_test)

                #calculating metrices
                mae, rmse, rsquared_value = self.evaluate_metric(y_test, predicted_values)

                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2_score", rsquared_value)

                # model registry does not work with file store
                # hence setting if else condition
                if tracking_url_type_store != "file":
                    #register the model
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")

                else:
                    mlflow.sklearn.log_model(model, "model")

                # before executing the training pipeline for experiment tracking and model registry
                # run these below code on terminal
                """
                export MLFLOW_TRACKING_URI=https://dagshub.com/Shreyanshu952/End_To_End_ML_Project.mlflow
                export MLFLOW_TRACKING_USERNAME=Shreyanshu952
                export MLFLOW_TRACKING_PASSWORD=e344a5dcf7202d9ed9fc310bdae64acda2b2d18b
                """
                # These commands are configuring MLflow to log and retrieve experiment information from the specified Dagshub tracking
                # server, and they include authentication credentials (username and password) for secure access to the tracking server.

                logging.info("Model Evaluation is done successfully")
                logging.info("Tracking Experiments and Model registry is successfully done")
        except Exception as e:
            logging.info("Error occured at model evaluation stage")
            raise customexception(e, sys)
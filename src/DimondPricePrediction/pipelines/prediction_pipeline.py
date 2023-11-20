import pandas as pd

import os
import sys

from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from src.DimondPricePrediction.utils.utils import load_object


# creating a class for custom data
class CustomData:

    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut
        self.color=color
        self.clarity=clarity

    # creating a definition for converting custom data into dataframe
    def get_data_into_dataframe(self):
        try:
            custom_data_input_dict = {"carat": [self.carat],
                                      "depth": [self.depth],
                                      "table": [self.table],
                                      "x": [self.x],
                                      "y": [self.y],
                                      "z": [self.z],
                                      "cut": [self.cut],
                                      "color": [self.color],
                                      "clarity": [self.clarity]}
        
            data = pd.DataFrame(custom_data_input_dict)
            logging.info("Our custom data has been converted into DataFrame")

            return data
    
        except Exception as e:
            logging.info("Exception occured during conversion of custom data into DataFrame")
            raise customexception(e, sys)


# creating class for prediction of custom data using our best model
class PredictPipeline:

    def __init__(self):
        pass
    
    # creating a definition for custom data prediction
    def predict(self, features):
        try:
            preprocessor_pickle_path = os.path.join("artifacts", "preprocessor.pkl")
            model_pickle_path = os.path.join("artifacts", "model.pkl")

            #loading the preprocessor and model pickle files
            preprocessor_pickle = load_object(preprocessor_pickle_path)
            model_pickle = load_object(model_pickle_path)
            logging.info("Preprocessor and Model pickle files loaded")

            # tranforming the input features (custom data)
            transformed_data = preprocessor_pickle.tranform(features)
            logging.info("Features has been tranformed")

            # predicting using our best model
            pred_data = model_pickle.predict(transformed_data)
            logging.info("Prediction done!")

            return pred_data
        
        except Exception as e:
            logging.info("Exception occured during custom data transformation and prediction")
            raise customexception(e, sys)
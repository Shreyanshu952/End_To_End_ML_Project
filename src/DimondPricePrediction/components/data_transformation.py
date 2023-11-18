import pandas as pd
import numpy as np

import os
import sys

from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.DimondPricePrediction.utils.utils import save_object

#creating the configuration of processor.pkl file
@dataclass
class DataTransformationConfig():
    processor_obj_file_path = os.path.join("artifacts", "processor.pkl")


# creating a class for data transformation
class DataTransformation:

    def __init__(self):
        #initiating the object for DataTransformationConfig
        self.data_tranformation_config  = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info("data transformation initiated")

            # Defining the numerical columns and categorical columns
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']

            # defining the custom rankings of the ordinal variable of the categorical columns
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("custom rankings of the ordinal variables has been defined")

            logging.info("numerical and categorical pipeline has been initiated")

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ("scaler", StandardScaler())
                ]
            )

            # tranforming and executing the above pipelines
            preprocessor = ColumnTransformer([
                ("num_feature", num_pipeline, numerical_cols),
                ("cat_feature", cat_pipeline, categorical_cols)
            ])

            logging.info("Pipelines has been created")

            return preprocessor

        except Exception as e:
            logging.info("Exception occured at get_data_tranformation stage")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:

            # reading csv files of train data and test data which has been provided as an output during data ingestion stage
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("reading of train and test data has been completed")
            logging.info(f"Train DataFrame head: \n {train_df.head().to_string()}")
            logging.info(f"Test DataFrame head: \n {test_df.head().to_string()}")

            #creating an object of get_data_tranformation
            preprocessor_obj = self.get_data_transformation()

            #separating independent and dependent features from train and test dataframe
            target_column_name = 'price'
            drop_columns = [target_column_name,'id']
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("independent and dependent variables are separated from train and test dataframe")

            #initializing the feature enginerring steps defined in the pipelines steps
            logging.info("Feature Engineering initialized")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Feature Engineering completed")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            #saving the preprocessing steps into pickle file
            save_object(
                file_path = self.data_tranformation_config.processor_obj_file_path, 
                obj = preprocessor_obj)
            
            logging.info("preprocessing pickle file has been saved")

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Exception occured at initialize_data_tranformation stage")
            raise customexception(e, sys)
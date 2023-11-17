import os
import sys

import pandas as pd

from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from src.DimondPricePrediction.components.data_ingestion import DataIngestion

# creating an object of DataIngestion class
obj = DataIngestion()

obj.initiate_data_ingestion()
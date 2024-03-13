import pandas as pd
import os
from Diamond_Prediction.components.data_ingestion import DataIngestion
from Diamond_Prediction.components.data_transformation import DataTransformation
from Diamond_Prediction.logger import logger



"""class training_pipeline_config():
    def __init__(self) -> None:
        pass 

class initiate_training():
    def __init__(self) -> None:
         pass"""
    
        
obj = DataIngestion()

train_data, test_data = obj.initiate_data_ingestion()

obj2 = DataTransformation()

xtrain, ytrain = obj2.initialize_data_transformation(train_data, test_data)


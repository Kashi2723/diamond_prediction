import pandas as pd
import os
from Diamond_Prediction.components.data_ingestion import DataIngestion
from Diamond_Prediction.components.data_transformation import DataTransformation
from Diamond_Prediction.components.model_trainer import ModelTrainer
from Diamond_Prediction.logger import logger



"""class training_pipeline_config():
    def __init__(self) -> None:
        pass 

class initiate_training():
    def __init__(self) -> None:
         pass"""
    
        
obj = DataIngestion()

train_data_path, test_data_path = obj.initiate_data_ingestion()

obj2 = DataTransformation()

train_trans_df, test_trans_df, train_target, test_target = obj2.initialize_data_transformation(train_data_path, test_data_path)

obj3 = ModelTrainer()

obj3.initiate_model_training(train_trans_df, test_trans_df, train_target, test_target)


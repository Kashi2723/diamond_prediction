import pandas as pd
import numpy as np
import os
from loguru import logger
from Diamond_Prediction.logger import logger
from dataclasses import dataclass
from Diamond_Prediction.utils.utils import save_object
from Diamond_Prediction.utils.utils import evaluate_model
from dotenv import find_dotenv

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 


@dataclass
class model_trainer_config():
    trained_model_file_path = os.path.join(os.path.dirname(find_dotenv()), "Artifacts",'model.pkl')
    
class ModelTrainer:
    @logger.catch
    def __init__(self):
        self.model_trainer_config = model_trainer_config()

    @logger.catch()
    def initiate_model_training(self,xtrain, xtest, ytrain, ytest):
        try:
            models = {
                'LinearRegression' : LinearRegression(),
                'Lasso' : Lasso(),
                'Ridge' : Ridge(),
                'ElasticNet' : ElasticNet()
            }

            model_report : dict = evaluate_model(xtrain, ytrain, xtest, ytest, models)
            print(model_report)
            print("="*100)
            logger.info(f"Model Report: {model_report}")

            # To get best model score in dictionary 
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            logger.info(best_model_score)

            best_model = models[best_model_name]

            print(f"Best Model Found , Model Name : {best_model_name}, R2 Score : {best_model_score}")
            print('='*100)
            logger.info(f"Best Model Found , Model Name : {best_model_name}, R2 Score : {best_model_score}")


            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

        except Exception as e:
            logger.exception(e)

            
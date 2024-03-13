import os 
import pickle 
import numpy as np
import pandas as pd
from Diamond_Prediction.logger import logger

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

@logger.catch()
def save_object(file_path,obj):
    try :
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logger.exception(e)


@logger.catch()
def evaluate_model(xtrain, ytrain, xtest, ytest, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train Model
            model.fit(xtrain, ytrain)

            ypred=model.predict(xtest)

            r2=r2_score(ytest, ypred)
            mae=mean_absolute_error(ytest, ypred)
            mse=mean_squared_error(ytest, ypred)
            rmse=np.sqrt(mse)

            report[list(models.keys())[i]]=r2
        return (report)

    except Exception as e:
        logger.exception(e)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logger.exception(e)
    
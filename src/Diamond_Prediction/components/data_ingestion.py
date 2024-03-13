import numpy as np
import pandas as pd
import os
from Diamond_Prediction.logger import logger
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from dotenv import find_dotenv

class DataIngestionConfig:
    raw_data_path = os.path.join(os.path.dirname(find_dotenv()),'Artifacts','raw_data.csv')
    train_data_path = os.path.join(os.path.dirname(find_dotenv()),'Artifacts','train_data.csv')
    test_data_path = os.path.join(os.path.dirname(find_dotenv()),'Artifacts','test_data.csv')


class DataIngestion:
    @logger.catch()
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        

    @logger.catch()    
    def initiate_data_ingestion(self):


        logger.info("data ingestion started")

        try:
            df = pd.read_csv(os.path.join(os.path.dirname(find_dotenv()),'notebooks','data','train.csv'))    
            logger.info("Read Dataset")


            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.success(f"Saved Raw Data Successully/n{df.head().to_string()}")



            logger.info('Performing Train test split')
            train_df,test_df = train_test_split(df,test_size=0.25,random_state=23)
            logger.info('Train Test splited')

        

            train_df.to_csv(self.ingestion_config.train_data_path, index = False )
            test_df.to_csv(self.ingestion_config.test_data_path , index = False)
            logger.success(f"Data Ingestion completed/n{df.head().to_string()}")

            return(self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            logger.exception(e)

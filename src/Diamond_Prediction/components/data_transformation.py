import os
import pandas as pd
import numpy as np

from dataclasses import dataclass
from Diamond_Prediction.logger import logger
from dotenv import find_dotenv

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from Diamond_Prediction.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(os.path.dirname(find_dotenv()), 'Artifacts','preprocessor.pk1')
    

class DataTransformation:
    @logger.catch()
    def __init__(self):
        self.data_transformation_Config = DataTransformationConfig()

    @logger.catch()
    def get_data_transformation(self):
        
        try :
            logger.info("Data Transformation initiated")

            # Define which columns should be ordinal-encoded and which should be scaled
            cat_col = ['cut','color','clarity']
            num_col = ['carot','depth','table','x','y','z']

            # Define the custom ranking for each variable :
            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logger.info("Pipeline Initiated")

            ## Numerical Pipeline 
            num_pipe = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )


            ## Categorical Pipeline
            cat_pipe = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy= 'most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers= [
                    ('num_pipeline', num_pipe,num_col),
                    ('cat_pipeline',cat_pipe,cat_col)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logger.exception(e)

    @logger.catch()
    def initialize_data_transformation(self, train_path, test_path):
        try :
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read Train and Test data")
            logger.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logger.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")

            preprocessor_obj = self.get_data_transformation()
                        
            target_column_name = 'price'
            drop_columns = [target_column_name,'id']
                        
            input_feature_train_df = train_df.drop(columns = drop_columns, axis = 1)
            input_feature_test_df = test_df.drop(columns= drop_columns, axis = 1)

            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]
                        
            input_feature_train = pd.DataFrame(preprocessor_obj.fit_transform(input_feature_train_df), columns = preprocessor_obj.get_feature_names_out())

            input_feature_test = pd.DataFrame(preprocessor_obj.transform(input_feature_test_df), columns = preprocessor_obj.get_feature_names_out())

            logger.info("Applying preprocessing object on training and testing datasets.")         

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_path,
                obj = preprocessor_obj
                
            )
            
            return(input_feature_train, input_feature_test)

        except Exception as e:
            logger.exception(e)

        
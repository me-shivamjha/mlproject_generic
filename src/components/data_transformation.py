import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # for handling categorical and numerical features and creating pipelines
from sklearn.impute import SimpleImputer # for handling missing values
from sklearn.pipeline import Pipeline # for creating pipelines
from sklearn.preprocessing import OneHotEncoder, StandardScaler # for handling categorical and numerical features

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass #decorator to automatically generate special methods like __init__() and __repr__() for the class
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl') #path to save the preprocessor object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() #create an instance of the DataTransformationConfig class

    def get_data_transformer_object(self):
        
        try:
            numerical_columns = ['writing score', 'reading score'] #list of numerical columns
            categorical_columns = ['gender', 
                                   'race/ethnicity', 
                                   'parental level of education', 
                                   'lunch', 
                                   'test preparation course'] #list of categorical columns
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')), #handle missing values in numerical columns using median strategy
                    ('scaler', StandardScaler()) #scale the numerical features using standard scaler
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), #handle missing values in categorical columns using most frequent strategy
                    ('one_hot_encoder', OneHotEncoder()), #encode categorical features using one-hot encoding
                    ("scaler", StandardScaler()) #scale the categorical features using standard scaler without centering
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}") #log the categorical columns
            logging.info(f"Numerical columns: {numerical_columns}") #log the numerical columns


            # combine the numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns), #apply the numerical pipeline to the numerical columns
                    ('cat_pipeline', cat_pipeline, categorical_columns) #apply the categorical pipeline to the categorical columns
                ]
            )

            return preprocessor #return the preprocessor object
        
        except Exception as e:
            raise CustomException(e, sys) #raise a custom exception if any error occurs during the creation of the preprocessor object
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path) #read the training data from the specified path
            test_df = pd.read_csv(test_path) #read the testing data from the specified path

            logging.info("Read train and test data completed") #log that the training and testing data has been read successfully

            logging.info("Obtaining preprocessing object") #log that the process of obtaining the preprocessor object has started

            preprocessor_obj = self.get_data_transformer_object() #get the preprocessor object

            target_column_name = 'math score' #name of the target column
            numerical_columns = ['writing score', 'reading score'] #list of numerical columns
            categorical_columns = ['gender', 
                                   'race/ethnicity', 
                                   'parental level of education', 
                                   'lunch', 
                                   'test preparation course'] #list of categorical columns
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1) #drop the target column from the training data to get the input features
            target_feature_train_df = train_df[target_column_name] #get the target column from the training data
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1) #drop the target column from the testing data to get the input features
            target_feature_test_df = test_df[target_column_name] #get the target column from the testing data

            logging.info("Applying preprocessing object on training and testing data") #log that the preprocessor object is being applied to the training and testing data  
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df) #fit and transform the input features of the training data using the preprocessor object
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df) #transform the input features of the testing data using the preprocessor object
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] #combine the input features and target column of the training data into a single array
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] #combine the input features and target column of the testing data into a single array    

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, #path to save the preprocessor object
                obj=preprocessor_obj #preprocessor object to be saved
            )


            return (
                train_arr, #return the transformed training data as an array
                test_arr, #return the transformed testing data as an array
                self.data_transformation_config.preprocessor_obj_file_path #return the path to save the preprocessor object
            )
        except Exception as e:
            raise CustomException(e, sys) #raise a custom exception if any error occurs during the data transformation process


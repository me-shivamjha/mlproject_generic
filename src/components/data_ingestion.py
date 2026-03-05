import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass #decorator to automatically generate special methods like __init__() and __repr__() for the class
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv') #path to save the training data
    test_data_path: str = os.path.join('artifacts', 'test.csv') #path to save the testing data
    raw_data_path: str = os.path.join('artifacts', 'data.csv') #path to save the raw data

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() #create an instance of the DataIngestionConfig class

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component") #log the start of the data ingestion process

        try:
            df = pd.read_csv(os.path.join('notebook\data\stud.csv')) #read the raw data from the specified path
            logging.info("Read the dataset as dataframe") #log that the dataset has been read successfully

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) #create the directory for saving the training data if it doesn't exist
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) #save the raw data to the specified path

            logging.info("Train test split initiated") #log that the train-test split process has started
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) #split the data into training and testing sets

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) #save the training data to the specified path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) #save the testing data to the specified path

            logging.info("Ingestion of the data is completed") #log that the data ingestion process has been completed

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            ) #return the paths to the training and testing data

        except Exception as e:
            raise CustomException(e, sys) #raise a custom exception if any error occurs during the data ingestion process
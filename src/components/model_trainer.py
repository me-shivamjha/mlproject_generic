# All the training code for the model goes here. This includes the training loop, validation loop, and any other necessary functions for training the model.
import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor, 
    RandomForestRegressor 
)
from sklearn.metrics import r2_score   
from xgboost import XGBRegressor    
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl') #path to save the trained model

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() #create an instance of the ModelTrainerConfig class

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data") #log the start of the model training process
            X_train, y_train = train_array[:,:-1], train_array[:,-1] #split the training data into features and target variable
            X_test, y_test = test_array[:,:-1], test_array[:,-1] #split the testing data into features and target variable

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "Adaboost Regressor": AdaBoostRegressor()
            } #dictionary of different regression models to be trained

            model_report: dict= evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models) #evaluate the models and get the report

            best_model_score = max(sorted(model_report.values())) #get the best R2 score from the report dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] #get the name of the best model from the report dictionary
            best_model = models[best_model_name] #get the best model from the models dictionary

            if best_model_score < 0.6: #if the best R2 score is less than 0.6, then raise an exception
                raise CustomException("No best model found") #raise a custom exception if no best model is found

            logging.info(f"Best found model on both training and testing dataset: {best_model_name}") #log the name of the best model

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            ) #save the best model to the specified path

            predicted = best_model.predict(X_test) #predict the target variable using the best model on the testing data
            r2_square = r2_score(y_test, predicted) #calculate the R2 score for the predictions
            return r2_square #return the R2 score for the predictions
        
        except Exception as e:
            raise CustomException(e, sys) #raise a custom exception if any error occurs during the model training process
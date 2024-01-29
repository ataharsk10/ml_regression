"""
03_model_trainer.py file takes preprocessed train and test data and
separate target colum from it and fit the model and
predict the test data set and
save the model
"""

import os
import sys
import argparse
from src.exception import CustomException
from src.logger import logging
from read_params import read_params
from src.utils import save_object,save_object_json,reg_evaluate_models
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
## Function for model training
def model_training(config_path):

    try:
        logging.info("Initiate model training")
        config = read_params(config_path)
        ## Reading preprocessed data path
        preprocessed_test_data_path = config["preprocessed_split_data"]["test_path"]
        preprocessed_train_data_path = config["preprocessed_split_data"]["train_path"]
        ## Define model parameters
        random_state = config["base"]["random_state"]
        n_estimators = config["estimators"]["RandomForest_regressor"]["params"]["n_estimators"]
        criterion = config["estimators"]["RandomForest_regressor"]["params"]["criterion"]
        max_depth = config["estimators"]["RandomForest_regressor"]["params"]["max_depth"]
        ## Reading preprocessed train and test dataset
        logging.info("Reading preprocessed train and test data")
        train_df = pd.read_csv(preprocessed_test_data_path)
        test_df = pd.read_csv(preprocessed_train_data_path)
        ## Separate target column from train and test dataframe
        logging.info("Separate target column from train and test data")
        X_train,y_train,X_test,y_test=(
            train_df.iloc[:,:-1],
            train_df.iloc[:,-1],
            test_df.iloc[:,:-1],
            test_df.iloc[:,-1]
            )
        ## Define model
        reg = RandomForestRegressor(
        n_estimators = n_estimators, 
        criterion = criterion,
        max_depth = max_depth, 
        random_state=random_state)
        ## Model fit
        logging.info("Model Fit")
        reg.fit(X_train, y_train)
        ## Prediction of test dataset
        logging.info("Predicting test data set")
        y_pred = reg.predict(X_test)
        ## Calling model evaluation function
        logging.info("Calling Model score")
        (mape, mae, mse, rmse, r2) = reg_evaluate_models(y_test, y_pred)
        #print(mape, mae, mse, rmse, r2)#

        ## Saving model object
        logging.info("Saving model object")
        model_dir = config["model_dir"]
        save_object(file_path=model_dir,obj=reg)

        ## Reading report path
        scores_file = config["reports"]["scores"]
        params_file = config["reports"]["params"]
        ## Saving scores.json
        scores = {
            "mape": mape,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
        save_object_json(file_path=scores_file,obj=scores)
        ## Saving params.json
        params = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth
            }
        save_object_json(file_path=params_file,obj=params)





    except Exception as e:
        raise CustomException(e,sys)




if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    ## Calling initiate_transformation
    model_training(config_path=parsed_args.config)
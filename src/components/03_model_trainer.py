import os
import sys
import argparse
from src.exception import CustomException
from src.logger import logging
from read_params import read_params
from src.utils import save_object,reg_evaluate_models
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


def model_training(config_path):

    try:
        logging.info("Initiate model training")
        config = read_params(config_path)
        preprocessed_test_data_path = config["preprocessed_split_data"]["test_path"]
        preprocessed_train_data_path = config["preprocessed_split_data"]["train_path"]
        
        random_state = config["base"]["random_state"]
        n_estimators = config["estimators"]["RandomForest_regressor"]["params"]["n_estimators"]
        criterion = config["estimators"]["RandomForest_regressor"]["params"]["criterion"]
        max_depth = config["estimators"]["RandomForest_regressor"]["params"]["max_depth"]
        
        logging.info("Reading preprocessed train and test data")
        train_df = pd.read_csv(preprocessed_test_data_path)
        test_df = pd.read_csv(preprocessed_train_data_path)

        logging.info("Separate target column from train and test data")
        X_train,y_train,X_test,y_test=(
            train_df.iloc[:,:-1],
            train_df.iloc[:,-1],
            test_df.iloc[:,:-1],
            test_df.iloc[:,-1]
            )
        
        reg = RandomForestRegressor(
        n_estimators = n_estimators, 
        criterion = criterion,
        max_depth = max_depth, 
        random_state=random_state)
        
        logging.info("Model Fit")
        reg.fit(X_train, y_train)

        logging.info("Predicting test data set")
        y_pred = reg.predict(X_test)

        logging.info("Calling Model score")
        (mape, mae, mse, rmse, r2) = reg_evaluate_models(y_test, y_pred)
        print(mape, mae, mse, rmse, r2)#

        ## Saving model object
        logging.info("Saving model object")
        model_dir = config["model_dir"]
        save_object(file_path=model_dir,obj=reg)

    except Exception as e:
        raise CustomException(e,sys)




if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    ## Calling initiate_transformation
    model_training(config_path=parsed_args.config)
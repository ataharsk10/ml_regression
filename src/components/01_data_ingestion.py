"""
data_ingestion.py file will connect to the external data source and
load the external data and save it to local and
split it as train and test set and again save it to local
"""

import os
import sys
import argparse
from src.exception import CustomException
from src.logger import logging
from read_params import read_params

import pandas as pd
from sklearn.model_selection import train_test_split

# Function for connecting external datasource
def get_data(config_path):
    config = read_params(config_path)
    try:
       logging.info("Connecting data source")
       data_path = config["data_source"]["DB_source"]
       df = pd.read_csv(data_path, sep=",", encoding='utf-8') # Write code to connect external datasource/DB
       return df 
    except Exception as e:
        raise CustomException(e,sys)
    
# Function for loading raw data and save it
def load_and_save(config_path):
    config = read_params(config_path)
    try:
        df = get_data(config_path)
        logging.info("Saving raw data")
        raw_data_path = config["load_data"]["raw_dataset_csv"]
        os.makedirs(os.path.dirname(raw_data_path),exist_ok=True)
        df.to_csv(raw_data_path, sep=",", index=False, header=True)
    except Exception as e:
        raise CustomException(e,sys)
    
    
# Function for splitting raw data into train, test set and save it
def split_and_saved_data(config_path):
    config = read_params(config_path)
    try:
        test_data_path = config["split_data"]["test_path"] 
        train_data_path = config["split_data"]["train_path"]
        raw_data_path = config["load_data"]["raw_dataset_csv"]
        split_ratio = config["split_data"]["test_size"]
        random_state = config["base"]["random_state"]

        df = pd.read_csv(raw_data_path, sep=",")

        #df = pd.read_csv(raw_data_path, sep=",")
        logging.info("Split Train-Test")
        train, test = train_test_split(
            df, 
            test_size=split_ratio, 
            random_state=random_state
            )
        
        logging.info("Saving train and test data")
        os.makedirs(os.path.dirname(train_data_path),exist_ok=True)
        train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
        os.makedirs(os.path.dirname(test_data_path),exist_ok=True)
        test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")
    except Exception as e:
        raise CustomException(e,sys)
    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    # Calling load_and_save function
    load_and_save(config_path=parsed_args.config)
    # Calling split and save function
    split_and_saved_data(config_path=parsed_args.config)
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
from src.utils import make_directory
import pandas as pd
from sklearn.model_selection import train_test_split

# Function for connecting external datasource
def get_data(config_path):
    try:
       config = config_path
       logging.info("Connecting data source")
       # Loading raw data path
       data_path = config["data_source"]["DB_source"]
       # Reading raw data from data path as dataframe
       df = pd.read_csv(data_path, sep=",", encoding='utf-8') # Write code to connect external datasource/DB
       # Returning raw dataframe
       return df 
    except Exception as e:
        raise CustomException(e,sys)
    
# Function for loading raw data and save it : it will call get_data function
def load_and_save(config_path):
    try:
        # Reading params.yaml file
        config = read_params(config_path)
        # Getting raw data
        df = get_data(config)
        logging.info("Saving raw data")
        # Location for saving raw data
        raw_data_path = config["load_data"]["raw_dataset_csv"]
        # Creating dir if not exist
        make_directory(raw_data_path)
        # Saving raw dataframe
        df.to_csv(raw_data_path, sep=",", index=False, header=True)
    except Exception as e:
        raise CustomException(e,sys)
    
    
# Function for splitting raw data into train, test set and save it
def split_and_saved_data(config_path):
    try:
        # Reading params.yaml file
        config = read_params(config_path)
        # Getting test data path, train data path, raw data path, split ratio, random state
        raw_data_path = config["load_data"]["raw_dataset_csv"]
        test_data_path = config["split_data"]["test_path"] 
        train_data_path = config["split_data"]["train_path"]
        split_ratio = config["split_data"]["test_size"]
        random_state = config["base"]["random_state"]
        # Reading raw dataframe
        df = pd.read_csv(raw_data_path, sep=",")
        # Perform train-test split
        logging.info("Split Train-Test")
        train, test = train_test_split(
            df, 
            test_size=split_ratio, 
            random_state=random_state
            )
        
        logging.info("Saving train and test data")
        # Create dir for train data if not exist  and save train dataframe
        make_directory(train_data_path)
        train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
        # Create dir for test data if not exist and save test dataframe
        make_directory(test_data_path)
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
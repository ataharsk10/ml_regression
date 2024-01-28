# Read params.yaml
import sys
import os
import yaml
import argparse
from src.exception import CustomException
from src.logger import logging

def read_params(config_path):
    try:
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    except Exception as e:
        raise CustomException(e,sys)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from src.exception import CustomException

# Function for making directory
def make_directory(file_path):
    os.makedirs(os.path.dirname(file_path),exist_ok=True)

# Function for save object
def save_object(file_path, obj):
    try:
        # Calling make_directory function
        make_directory(file_path)
        # Saving object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
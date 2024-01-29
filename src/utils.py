import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from src.exception import CustomException
## Regression Scores
from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             mean_squared_error,
                             mean_absolute_percentage_error
                            )

## Function for making directory
def make_directory(file_path):
    os.makedirs(os.path.dirname(file_path),exist_ok=True)

## Function for save object
def save_object(file_path, obj):
    try:
        ## Calling make_directory function
        make_directory(file_path)
        ## Saving object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

## Function for evaluating regression scores  
def reg_evaluate_models(true,predicted):
    try:
       ## Mean absolute percentage error
        mape = mean_absolute_percentage_error(true, predicted)
        ## Mean absolute error
        mae = mean_absolute_error(true, predicted)
        ## Mean squared error
        mse = mean_squared_error(true, predicted)
        ## Root mean squared error
        rmse = np.sqrt(mse)
        ## R Squared
        r2 = r2_score(true, predicted)
        ## Adjusted R Squared
        #a_r2 = 1 - (1-r2)*(len(X_test)-1)/(len(X_test)-X_test.shape[1]-1)
        return mape, mae, mse, rmse, r2


    except Exception as e:
        raise CustomException(e, sys)
    
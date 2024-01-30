""""
data_transformation.py will take train.csv and test.csv and
apply data preprocessing/transformation over train,test file and
save it as object: preprocessor.pkl and
return train, test array
"""

import os
import sys
import argparse
from src.exception import CustomException
from src.logger import logging
from read_params import read_params
from src.utils import save_object,make_directory
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

def initiate_data_transformer_object(config):
    config = config
    try:
        ## Reading numerical and categorical column
        numerical_columns = config["base"]["numerical_columns"]
        categorical_columns = config["base"]["categorical_columns"] 
        ## Numerical columns: Creating pipeline for Imputations and Scaling
        num_pipeline= Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
                )

        ## Categorical columns: Creating pipeline for Imputations, Encoding and Scaling
        cat_pipeline=Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
                )
        logging.info(f"Categorical columns: {categorical_columns}")
        logging.info(f"Numerical columns: {numerical_columns}")
        
        ## Creating column transformer for above pipelines
        preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]
            )
        ## Returning preprocessor object
        return preprocessor
        
    except Exception as e:
        raise CustomException(e,sys)

def initiate_transformation(config_path):
    config = read_params(config_path)
    try:
        ## Read train test data file path
        test_data_path = config["split_data"]["test_path"] 
        train_data_path = config["split_data"]["train_path"]
        ## Read train test dataframe
        train_df=pd.read_csv(train_data_path)
        test_df=pd.read_csv(test_data_path)
        logging.info("Read train and test data completed")
        ## Calling preprocessing_obj function
        logging.info("Obtaining preprocessing object")
        preprocessing_obj= initiate_data_transformer_object(config)
        ## Reading target column
        target_column = config["base"]["target_col"]
        ## Dropping target column form training data set
        feature_train_df = train_df.drop(columns=[target_column],axis=1)
        target_train_df = train_df[target_column]
        ## Dropping target column from test data set
        feature_test_df = test_df.drop(columns=[target_column],axis=1)
        target_test_df = test_df[target_column]
        ## Applying preprocessing_obj in train and test data set
        logging.info(f"Applying preprocessing object on training and testing dataframe.")
        feature_train_arr=preprocessing_obj.fit_transform(feature_train_df)
        feature_test_arr=preprocessing_obj.transform(feature_test_df)
        ## Merging train and test dataset with corresponding target column
        train_arr = np.c_[feature_train_arr, np.array(target_train_df)]
        test_arr = np.c_[feature_test_arr, np.array(target_test_df)]
        ## Saving preprocessing object
        logging.info(f"Saved preprocessing object.")
        preprocessor_obj_file_path = config['preprocessor_obj_path']
        save_object(file_path=preprocessor_obj_file_path,obj=preprocessing_obj)
        
        ## Saving transform train, test dataset
        ## Train dataset
        logging.info("Saving training data after preprocessing")
        ## Location for saving Preprocessed training data
        preprocessed_train_data_path = config["preprocessed_split_data"]["train_path"]
        ## Creating dir if not exist
        make_directory(preprocessed_train_data_path)
        ## Saving raw dataframe
        df_train = pd.DataFrame(train_arr)
        df_train.to_csv(preprocessed_train_data_path, sep=",", index=False, header=False)

        ## Test dataset
        logging.info("Saving testing data after preprocessing")
        ## Location for saving Preprocessed training data
        preprocessed_test_data_path = config["preprocessed_split_data"]["test_path"]
        ## Creating dir if not exist
        make_directory(preprocessed_test_data_path)
        ## Saving raw dataframe
        df_test = pd.DataFrame(test_arr)
        df_train.to_csv(preprocessed_test_data_path, sep=",", index=False, header=False)

        ## Returning train, test array and preprocessor_obj_file_path
        """
        return (
                train_arr,
                test_arr,
                preprocessor_obj_file_path
            )
        """

    except Exception as e:
        raise CustomException(e,sys)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    ## Calling initiate_transformation
    initiate_transformation(config_path=parsed_args.config)
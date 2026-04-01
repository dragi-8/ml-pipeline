import nltk
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

log_dir="logs"
dir=os.makedirs(log_dir, exist_ok=True)
logger=logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)
logger.propagate = False

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_path=os.path.join(log_dir,"data_ingestion.log")
file_handler=logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load the dataset from the specified file path.

    Args:
        file_path (str): The path to the dataset file."""
    
    try:
        df= pd.read_csv(data_url)
        logger.debug(f"Data loaded successfully from {data_url}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame,) -> pd.DataFrame:
    """
    Preprocess the dataset by handling missing values and encoding categorical variables.

    Args:
        df (pd.DataFrame): The input dataset."""
    
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
        logger.debug("Data preprocessing completed successfully")

        return df
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame,file_loc:str) -> None:
    """
    Save the preprocessed training and testing datasets to the specified file paths.

    Args:
        train_df (pd.DataFrame): The preprocessed training dataset.
        test_df (pd.DataFrame): The preprocessed testing dataset.
        train_path (str): The path to save the training dataset.
        test_path (str): The path to save the testing dataset."""
    
    try:
        os.makedirs(file_loc, exist_ok=True)
        train_path=os.path.join(file_loc,"train.csv")
        test_path=os.path.join(file_loc,"test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.debug(f"Data saved successfully to {train_path} and {test_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


def main():
    try:
        test_size=0.2
        data_url="https://raw.githubusercontent.com/dragi-8/ml-pipeline/main/spam.csv"
        
        logger.debug(f"Starting data ingestion with data_url={data_url}")
        df=load_data(data_url)
        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=2)
        save_data(train_data,test_data,'./data')
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__=="__main__":
    main()    
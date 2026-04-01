import numpy as np
import pandas as pd
import logging
#import tfidf_vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os


log_dir="logs"
dir=os.makedirs(log_dir, exist_ok=True)
logger=logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)
logger.propagate = False

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_path=os.path.join(log_dir,"feature_engineering.log")
file_handler=logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path:str) -> pd.DataFrame:
    """
    Load the dataset from the specified file path.

    Args:
        file_path (str): The path to the dataset file."""
    
    try:
        df= pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def  apply_tfidf_vectorization(train_df: pd.DataFrame, test_df: pd.DataFrame,max_features: int) -> tuple:
    """
    Apply TF-IDF vectorization to the text data in the training and testing datasets.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The testing dataset.
        text_column (str): The name of the column containing the text data."""
    
    try:
        tfidf=TfidfVectorizer(max_features=max_features)
        X_train=train_df["text"].fillna("").values
        X_test=test_df["text"].fillna("").values
        y_train=train_df["label"].values
        y_test=test_df["label"].values


        x_train_bow=tfidf.fit_transform(X_train)
        x_test_bow=tfidf.transform(X_test)

        
        test_tfidf=pd.DataFrame(x_test_bow.toarray())
        test_tfidf['label']=y_test
    
        train_tfidf=pd.DataFrame(x_train_bow.toarray())
        train_tfidf['label']=y_train   
        
        logger.debug("TF-IDF vectorization applied successfully")
        return train_tfidf, test_tfidf
    except Exception as e:
        logger.error(f"Error applying TF-IDF vectorization: {e}")
        raise    


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the transformed dataset to the specified file path.

    Args:
        df (pd.DataFrame): The transformed dataset.
        file_path (str): The path to save the dataset."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise


def main():
    try:
        train_file="data/train_preprocessed.csv"
        test_file="data/test_preprocessed.csv"
        logger.debug(f"Starting feature engineering with train_file={train_file} and test_file={test_file}")
        train_df=load_data(train_file)
        test_df=load_data(test_file)
        x_train_tfidf, x_test_tfidf=apply_tfidf_vectorization(train_df, test_df, max_features=50)
        save_data(pd.DataFrame(x_train_tfidf), os.path.join('data','processed','train_tfidf.csv'))
        save_data(pd.DataFrame(x_test_tfidf), os.path.join('data','processed','test_tfidf.csv'))
        logger.debug("Feature engineering completed and saved successfully")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
        


if __name__=="__main__":
    main()        
import numpy as np 
import pandas as pd
import logging
#import random_forest_classifier
from sklearn.ensemble import RandomForestClassifier
import os
import pickle


log_dir="logs"
dir=os.makedirs(log_dir, exist_ok=True)
logger=logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)
logger.propagate = False

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_path=os.path.join(log_dir,"model_building.log")
file_handler=logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def train_random_forest_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame,n_estimators: int,random_state: int) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier on the training dataset and evaluate it on the testing dataset.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The testing dataset."""
    
    try:
        X_train=train_df.drop(columns=['label']).values
        y_train=train_df['label'].values
        X_test=test_df.drop(columns=['label']).values
        y_test=test_df['label'].values
        
        rf_classifier=RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf_classifier.fit(X_train, y_train)
        
        logger.debug("Random Forest Classifier trained successfully")
        return rf_classifier
    except Exception as e:
        logger.error(f"Error training Random Forest Classifier: {e}")
        raise


def save_model(model: RandomForestClassifier, file_path: str) -> None:
    """
    Save the trained model to the specified file path.

    Args:
        model (RandomForestClassifier): The trained model to be saved.
        file_path (str): The path where the model will be saved."""
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error saving model at {file_path}: {e}")
        raise

def main():
    try:
        metrics={'nestimators':100,'random_state':42}
        train_df=pd.read_csv("data/processed/train_tfidf.csv")
        test_df=pd.read_csv("data/processed/test_tfidf.csv")
        clf=train_random_forest_classifier(train_df,test_df,metrics['nestimators'],metrics['random_state'])  

        save_model(clf,"models/clf_model.pkl")
        logger.debug("Model training and saving completed successfully")  
    except Exception as e:    
        logger.error(f"Error in model training and saving: {e}")
        raise

if __name__=="__main__":    
    main()
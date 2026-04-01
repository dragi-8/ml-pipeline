import numpy as np 
import json
import pandas as pd
import logging
#import random_forest_classifier
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
#import evaluation_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



log_dir="logs"
dir=os.makedirs(log_dir, exist_ok=True)
logger=logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
logger.propagate = False

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_path=os.path.join(log_dir,"model_evaluation.log")
file_handler=logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path: str) -> RandomForestClassifier:
    """
    Load the trained model from the specified file path.

    Args:
        file_path (str): The path to the saved model file."""
    
    try:
        with open(file_path, 'rb') as f:
            model=pickle.load(f)
        logger.debug(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise

def evaluate_model(model: RandomForestClassifier, test_df: pd.DataFrame) -> dict:    
    """
    Evaluate the trained model on the testing dataset and return the evaluation metrics.

    Args:
        model (RandomForestClassifier): The trained model to be evaluated.
        test_df (pd.DataFrame): The testing dataset."""
    
    try:
        X_test=test_df.drop(columns=['label']).values
        y_test=test_df['label'].values
        
        y_pred=model.predict(X_test)
        
        accuracy=accuracy_score(y_test, y_pred)
        precision=precision_score(y_test, y_pred, average='weighted')
        recall=recall_score(y_test, y_pred, average='weighted')
        f1=f1_score(y_test, y_pred, average='weighted')
        metrics={"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
        logger.debug("Model evaluation completed successfully")
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def save_evaluation_metrics(metrics: dict, file_path: str) -> None:
    """
    Save the evaluation metrics to a JSON file at the specified file path.

    Args:
        metrics (dict): The evaluation metrics to be saved.
        file_path (str): The path where the evaluation metrics will be saved."""
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug(f"Evaluation metrics saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error saving evaluation metrics to {file_path}: {e}")
        raise


def main():
    try:
        model=load_model("models/clf_model.pkl")
        test_df=pd.read_csv("data/processed/test_tfidf.csv")
        metrics=evaluate_model(model, test_df)
        save_evaluation_metrics(metrics, "metrics/metrics.json")
        logger.debug("Model evaluation and saving metrics completed successfully")
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise   


if __name__=="__main__":
    main()     

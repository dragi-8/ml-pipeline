import numpy as np
import pandas as pd
import logging
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#import label_encoder
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
import string
import os



log_dir="logs"
dir=os.makedirs(log_dir, exist_ok=True)
logger=logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)
logger.propagate = False

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_path=os.path.join(log_dir,"data_preprocessing.log")
file_handler=logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transform the input text by removing stop words and applying stemming.

    Args:
        text (str): The input text to be transformed."""
    try:
        ps= PorterStemmer()
        text=text.lower()
        text=nltk.word_tokenize(text)
        text=[word for word in text if word not in stopwords.words("english") and word not in string.punctuation]    
        text=[ps.stem(word) for word in text]
        text=" ".join(text)
        return text
    except Exception as e:
        logger.error(f"Error occurred while transforming text: {e}")
        raise


def preprocess_data(df: pd.DataFrame,text_column: str,target_column: str) -> pd.DataFrame:
    """
    Preprocess the dataset by applying text transformation and encoding labels.

    Args:
        df (pd.DataFrame): The input dataset.
        text_column (str): The name of the column containing the text data.
        target_column (str): The name of the column containing the target labels."""
    try:
        df["text"]=df["text"].apply(transform_text)
        le=LabelEncoder()
        df["label"]=le.fit_transform(df["label"])
        #now drop duplicates    
        df.drop_duplicates(inplace=True)
        logger.debug("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.error(f"Error occurred during data preprocessing: {e}")
        raise    

def main():
    try:
       train_data="data/train.csv"
       test_data="data/test.csv"
       logger.debug(f"Starting data preprocessing with train_data={train_data}")
       train_df=pd.read_csv(train_data)
       test_df=pd.read_csv(test_data)
       train_df=preprocess_data(train_df,"text","label")
       test_df=preprocess_data(test_df,"text","label")
       os.makedirs("data", exist_ok=True)
       train_df.to_csv("data/train_preprocessed.csv", index=False)
       test_df.to_csv("data/test_preprocessed.csv", index=False)
       logger.debug("Data preprocessing completed and saved successfully")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
    
if __name__=="__main__":
    main()


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def load_data(file):
    try:
        df = pd.read_csv(file , header=0 , names = ['ID','Topic','Sentiment','Text'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_text(text) : 
    if not isinstance(text , str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
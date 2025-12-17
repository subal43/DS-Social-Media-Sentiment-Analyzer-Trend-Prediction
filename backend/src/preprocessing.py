import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import datetime
nltk.download('punkt_tab')


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


def preprocess_text(text):
    if not isinstance(text , str):
        return ""
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    token = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in token if word not in stop_words]
    return ' '.join(cleaned_tokens)
    
def prepare_data(df):
    df = df.dropna(subset = ['Text'])
    df.loc[:,'Cleaned_text'] = df['Text'].apply(clean_text)
    df.loc[:,'Processed_text'] = df['Cleaned_text'].apply(preprocess_text)
    return df

def add_dates(df):
    start_date = datetime.date.today() - datetime.timedelta(days=365)
    end_date = datetime.date.today()
    days_range = (end_date - start_date).days
    random_days = np.random.randint(0, days_range, size=len(df))
    df['Date'] = [start_date + datetime.timedelta(days=int(day)) for day in random_days]
    df['Date'] = pd.to_datetime(df['Date'])
    return df
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score , classification_report

MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..' , 'models')
if not os.path.exists(MODEL_FILE):
    os.makedirs(MODEL_FILE)

MODEL_PATH = os.path.join(MODEL_FILE, 'sentiment_model.pkl')

def train_model(df):    
    x = df['Processed_text']
    y = df['Sentiment']
    x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000 , multi_class = 'ovr'))
    ])


    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, accuracy, report


def save_model(model):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        return None
   
def predict_sentiment(model, texts):
    return model.predict(texts)[0]
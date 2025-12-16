import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..' , 'models')
if not os.path.exists(MODEL_FILE):
    os.makedirs(MODEL_FILE)

MODEL_PATH = os.path.join(MODEL_FILE, 'sentiment_model.pkl')


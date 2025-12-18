import streamlit as st
import pandas as pd
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.preprocessing import load_data, prepare_data, add_dates
from src.model import train_model, save_model, load_model, predict_sentiment

st.set_page_config(
    page_title="Social Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def get_data():
    if os.path.exists('data/twitter_training.csv'):
        data_path = 'data/twitter_training.csv'
    
    df = load_data(data_path)
    if not df is None:
        df = prepare_data(df)
        df = add_dates(df)
    return df

@st.cache_resource
def get_model(df=None):
    model = load_model('models/sentiment_model.pkl')
    if model is None and df is not None :
        sample_size = min(len(df),5000)
        model = train_model(df.sample(sample_size, random_state=42))
        save_model(model, 'models/sentiment_model.pkl')
        return model
    else:
        return model
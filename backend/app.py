import streamlit as st
import pandas as pd
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.preprocessing import load_data, prepare_data, add_dates
# from src.model import train_model, save_model, load_model, predict_sentiment

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
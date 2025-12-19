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
def get_model():
    return load_model('models/sentiment_model.pkl')

   
def ensure_model(model, df):
    if model is None and df is not None:
        sample_size = min(len(df), 5000)
        model = train_model(df.sample(sample_size, random_state=42))
        save_model(model, 'models/sentiment_model.pkl')
        st.cache_resource.clear()
        model = get_model()

    return model

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sentiment Analyzer", "Dashboard", "Trend Analysis"])

with st.spinner('Loading data...'):
    data = get_data()

if data.empty:
    st.error("No data available. Please ensure the data file exists.")
    st.stop()

model = get_model()
model = ensure_model(model, data)



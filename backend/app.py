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


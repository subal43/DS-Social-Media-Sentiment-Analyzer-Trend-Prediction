import streamlit as st
import pandas as pd
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.preprocessing import load_data, prepare_data, add_dates,clean_text, preprocess_text
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
    return load_model()

   
def ensure_model(model, df):
    if model is None and df is not None:
        # sample_size = min(len(df), 5000)
        model = train_model(df)
        save_model(model)
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

if page == "Home":
    st.title("Welcome to the Social Sentiment Analyzer")

    st.markdown("""
    Welcome to the **Social Media Sentiment Analyzer & Trend Prediction** Dashboard.
    
    This tool allows you to:
    - **Analyze Sentiment:** Predict if a text is Positive, Negative, or Neutral.
    - **Visualize Data:** Explore the dataset with interactive charts.
    - **Track Trends:** See how sentiment changes over time.
    Navigate through the sidebar to access different features.
    """)
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", use_container_width=True)

elif page == "Sentiment Analyzer":
    st.header("Sentiment Analyzer")
    st.markdown("Enter text below to analyze its sentiment.")
    user_input = st.text_area("Enter your text here:", height=200)
    if st.button("Analyze"):
        if user_input:
            cleaned = clean_text(user_input)
            processed = preprocess_text(cleaned)
            prediction = predict_sentiment(model,processed)
            st.subheader(f"Predicted Sentiment : {prediction}")
            if prediction == 'Positive':
                st.success("😊 This is a Positive sentiment!")
            elif prediction == 'Negative':
                st.error("☹️ This is a Negative sentiment!")
            else:
                st.info("😐 This is a Neutral sentiment!")
        else:
            st.warning("Please enter some text to analyze its sentiment.")

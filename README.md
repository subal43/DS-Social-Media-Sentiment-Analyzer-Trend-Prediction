# Social Media Sentiment Analyzer & Trend Prediction

## 📊 Project Overview
The **Social Media Sentiment Analyzer & Trend Prediction** is a data science project designed to analyze the sentiment of social media text (Positive, Negative, Neutral) and visualize trends over time. Built with Python and Streamlit, it offers an interactive dashboard for checking sentiment on the fly and exploring historical data trends.

## 🚀 Features
- **Sentiment Analysis**: Real-time prediction of text sentiment using a Machine Learning pipeline (Logistic Regression with TF-IDF).
- **Interactive Dashboard**: Visualize sentiment distribution and explored raw data.
- **Trend Analysis**: dynamic line charts to track sentiment changes over daily, weekly, or monthly periods.
- **Data Preprocessing**: Robust text cleaning (removing URLs, special characters) and lemmatization.


## 🛠️ Tech Stack
- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Programming Language**: Pyton 3.x
- **Machine Learning**: Scikit-Learn (Logistic Regression, TF-IDF Vectorizer)
- **Data Manipulation**: Pandas, NumPy
- **NLP**: NLTK (Stopwords, WordNet Lemmatizer)
- **Visualization**: Plotly Express


## 📂 Project Structure
```
d:\sentiment-analyzer\
├── backend\
│   ├── app.py              # Main Streamlit application entry point
│   └── src\
│       ├── model.py        # ML model training, saving, and prediction logic
│       ├── preprocessing.py# Text cleaning and data preparation functions
│       └── trend.py        # Trend analysis and visualization logic
├── data\                   # Directory for dataset (twitter_training.csv)
├── models\                 # Directory where the trained model (sentiment_model.pkl) is saved
├── ran_app.bat             # Batch script to easily run the application on Windows
└── README.md               # Project documentation
```

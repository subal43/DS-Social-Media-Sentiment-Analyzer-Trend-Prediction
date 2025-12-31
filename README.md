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
## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd sentiment-analyzer
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Since there is no `requirements.txt` provided, install the necessary packages manually:
    ```bash
    pip install streamlit pandas numpy scikit-learn nltk plotly
    ```

4.  **Download NLTK Data**
    The app requires NLTK data. You can download it via python or run the app once, as the code attempts to download `punkt_tab` automatically. You may also need:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## 🏃 Usage
### Running with Batch Script (Windows)
Simply double-click the `ran_app.bat` file in the root directory.

### Running Manually
Execute the following command in the terminal from the project root:
```bash
streamlit run backend/app.py
```


## 🧠 Model Details
- The project uses a **Logistic Regression** model wrapped in a **OneVsRestClassifier**.
- Text is vectorized using **TF-IDF**.
- The model is trained on the first run if not present and saved to `models/sentiment_model.pkl`.


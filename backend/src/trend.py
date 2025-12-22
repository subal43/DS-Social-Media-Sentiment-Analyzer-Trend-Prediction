import streamlit as st
import plotly.express as px


def sentiment_trend_analysis(data):
    """
    Displays interactive sentiment trend analysis with:
    - Daily / Weekly / Monthly aggregation
    - Separate charts per sentiment
    - Optional moving average smoothing
    """

    st.subheader("Sentiment Trend Analysis")

   
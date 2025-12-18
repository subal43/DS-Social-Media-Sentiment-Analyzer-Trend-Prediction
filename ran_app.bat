@echo off
echo Starting Social Media Sentiment Analyzer...
cd /d "%~dp0"
streamlit run backend/app.py
pause

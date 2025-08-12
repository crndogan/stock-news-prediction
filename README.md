# Stock News and Market Movement Prediction

This repository contains an end-to-end system that predicts daily S&P 500 market direction using financial and political news headlines. The project combines natural language processing (NLP), machine learning, and automation to explore how the sentiment and emotional tone of news headlines may relate to next-day market movement.

All data is collected, processed, and modeled automatically using Python scripts and GitHub Actions.

## Project Summary

This project was developed as part of a capstone focused on using unstructured news text to make structured financial predictions. Every day, the system scrapes fresh news headlines, calculates sentiment and emotion scores, and merges the data with historical S&P 500 performance. A machine learning model (Random Forest) is then used to predict whether the market will go up or down the following day.

The prediction and analysis are published on a public dashboard, making the results accessible for investors, analysts, and students.

## Live Dashboard

View the latest prediction and trends:  
[https://stock-news-headlines-prediction.streamlit.app/](https://stock-news-headlines-prediction.streamlit.app/)

The dashboard includes:

- Next trading day prediction and model confidence
- Historical accuracy and performance metrics
- Sentiment and emotion score trends
- Word clouds highlighting topics associated with market up/down days
- Table of recent S&P 500 closing data

## GitHub Actions: Daily Automation

This repository uses GitHub Actions to automate the full data pipeline. The workflow is scheduled to run every day at 11:00 PM Pacific Time. It can also be triggered manually.

The GitHub Action performs the following steps:

1. Scrapes daily financial and political headlines using the NewsData.io API.
2. Downloads S&P 500 data using the yFinance library.
3. Cleans and processes the headlines (text normalization, punctuation removal, etc.).
4. Applies sentiment scoring (VADER) and emotion detection (NRCLex).
5. Merges news and stock data to create a labeled dataset.
6. Runs a Random Forest model to make a new prediction.
7. Updates all relevant `.csv` and `.xlsx` files used by the Streamlit app.

All scripts are version-controlled, and the system is designed to be robust, reproducible, and extensible.

## Repository Structure

├── .github/workflows/ # GitHub Actions for daily automation
├── notebooks/ # Jupyter Notebooks for scraping, analysis, modeling
├── streamlit_app.py # Streamlit dashboard code
├── stock_news_tone.xlsx # Daily sentiment and emotion scores
├── sp500_cleaned.csv # S&P 500 index data
├── prediction_results.csv # Prediction history (actual vs. predicted)
├── tomorrow_prediction.csv # Most recent prediction
├── topic_modeling.csv # Topic clusters per day
├── topic_up_down.csv # Keywords for up/down days
├── requirements.txt # Python dependencies
└── README.md # Project documentation (this file)


## Technologies Used

- Python (pandas, numpy, scikit-learn)
- VADER Sentiment Scoring
- NRCLex Emotion Detection
- LDA Topic Modeling
- Streamlit (Dashboard)
- GitHub Actions (Automation)
- NewsData.io (News API)
- yFinance (Market Data)

## Use Cases

This project is designed for:

- Data analysts and students exploring NLP in finance
- Investors interested in daily headline-based market sentiment
- Educators and researchers looking for applied machine learning workflows

## License

This project is for educational and academic purposes.



 

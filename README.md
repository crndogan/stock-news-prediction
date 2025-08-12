# Stock News and Market Movement Prediction

This project explores how the sentiment and content of daily financial and political news headlines can help predict short-term movements in the S&P 500 index. It uses natural language processing (NLP) and machine learning to classify whether the market is likely to move up or down the next trading day.

By automating the collection and analysis of news data, this system provides a structured way to evaluate how headline tone and emotion may relate to market behavior.

## Project Objectives

- Automatically collect daily financial and political headlines
- Gather S&P 500 market data for the same time period
- Clean and process headlines using standard NLP techniques
- Apply sentiment scoring (VADER) and emotion detection (NRCLex)
- Use topic modeling to group related news themes
- Build a labeled dataset matching news sentiment with next-day market movement
- Train a classification model (Random Forest) to predict future market direction
- Share results via a web-based dashboard

## Live Dashboard

Access the prediction dashboard at the following link:  
[https://stock-news-headlines-prediction.streamlit.app/](https://stock-news-headlines-prediction.streamlit.app/)

The dashboard includes:

- Daily market movement predictions and confidence levels
- Charts comparing predicted vs. actual market outcomes
- Sentiment and emotion trends across days
- Word clouds showing top topics during positive and negative market days
- Historical stock close data for the S&P 500

## Automation with GitHub Actions

This project includes a GitHub Actions workflow that automates the following tasks each day:

1. Scrapes daily headlines from the NewsData.io API
2. Downloads the most recent S&P 500 stock data using yFinance
3. Cleans and preprocesses headlines for analysis
4. Applies sentiment scoring and emotion tagging
5. Merges news and stock data to update the training dataset
6. Runs the trained model to generate a new prediction
7. Saves results to CSV files used by the Streamlit dashboard

The workflow is scheduled to run nightly at 11:00 PM Pacific Time. It can also be triggered manually to test or refresh the system.

## Tools and Technologies

- Python (pandas, numpy, scikit-learn)
- NLP Libraries: VADER, NRCLex
- Topic Modeling: LDA
- Data Sources: NewsData.io, yFinance
- Dashboard: Streamlit
- Automation: GitHub Actions

## Intended Users

This project is designed for:

- Data analysts and scientists exploring financial NLP applications
- Students learning how to build end-to-end data pipelines
- Business users interested in how news sentiment may relate to market behavior
- Researchers or instructors looking for real-world, reproducible use cases

## Repository Structure

- `notebooks/` – Jupyter Notebooks for data collection, preprocessing, and modeling
- `streamlit_app.py` – Streamlit dashboard application script
- `*.csv` / `*.xlsx` – Data files generated and used by the model
- `requirements.txt` – Python package dependencies
- `.github/workflows/` – Automation scripts for daily data updates

## Summary

This project demonstrates how unstructured news data can be converted into structured insights using NLP and machine learning. It offers a repeatable process for building daily financial predictions, and a flexible dashboard for visualizing results. The system is fully automated, version-controlled, and designed for educational and exploratory purposes.


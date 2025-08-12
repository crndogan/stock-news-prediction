# 📈 Stock News & Market Movement Prediction

This project helps us understand how news headlines affect the stock market. It uses machine learning and natural language processing (NLP) to predict if the **S&P 500 index** will go **up or down** the next day based on the **tone of financial and political news**.

🧠 The system turns headlines into data, scores their **sentiment** and **emotions**, then uses a model to make daily predictions.

## 🔍 What This Project Does

- Collects financial and political news headlines every day using the **NewsData.io API**
- Gets S&P 500 stock prices using the **yFinance** Python library
- Cleans the news text (lowercase, punctuation removal, stopwords)
- Calculates:
  - **Sentiment scores** using VADER (positive, negative, compound)
  - **Emotions** using NRCLex (like fear, trust, joy, anger)
  - **Topics** using LDA topic modeling
- Labels each day with **"Up" or "Down"** based on next-day market movement
- Trains a **Random Forest** machine learning model
- Predicts whether the market will go up or down tomorrow
- Shares the results in an easy-to-use dashboard

## 📊 Live Dashboard

Check out the daily predictions and sentiment analysis here:  
👉 [https://stock-news-headlines-prediction.streamlit.app/](https://stock-news-headlines-prediction.streamlit.app/)

The dashboard shows:

- Next-day market prediction with confidence score
- Historical performance chart (actual vs predicted)
- Sentiment and emotion trends
- Word clouds of popular topics on up/down market days
- Table of S&P 500 market close data

## ⚙️ GitHub Actions Automation

The GitHub Action runs **every night** at 11:00 PM Pacific Time. It:

1. Scrapes the latest headlines from NewsData.io
2. Downloads the newest S&P 500 stock prices
3. Cleans and scores the news
4. Matches each day’s news with the next trading day's stock movement
5. Makes a prediction using the machine learning model
6. Saves the updated results to `.csv` files used in the dashboard

✅ You can also **manually trigger** the workflow to refresh data or test updates.

All code, data, and model files are stored and versioned on this GitHub repo.

## 🛠️ Tools Used

- Python
- Pandas, NumPy, scikit-learn
- VADER Sentiment
- NRCLex Emotion Detection
- LDA Topic Modeling
- Streamlit (for dashboard)
- GitHub Actions (for automation)
- yFinance & NewsData.io APIs

## 🙋‍♀️ Who Is This For?

- Students learning data science and NLP
- Analysts interested in connecting news to market movement
- Investors who want to explore headline-based prediction
- Instructors or researchers looking for real-world ML projects

---

📂 All code is open-source. Feel free to use or extend it for your own learning or research.

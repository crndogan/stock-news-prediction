This project studies how the tone of financial and political news headlines affects daily stock market movements, focusing on the S&P 500 index. Many investors make decisions based on headlines, but this system adds structure by using Natural Language Processing (NLP) and machine learning to turn news into data-driven insights.

The system collects news headlines daily using the NewsData.io API and gathers S&P 500 data via yFinance. Headlines are cleaned, scored for sentiment (VADER), and emotions (NRCLex), and grouped into topics (LDA). A Random Forest model is trained on this data to predict if the market will go up or down the next day. Results are shared in a public dashboard built with Streamlit.

🔗 Live Dashboard

⚙️ GitHub Actions Workflow
This project includes a scheduled GitHub Actions workflow that:

Automatically scrapes new financial and political headlines daily (via NewsData.io).

Downloads the latest S&P 500 stock prices (via yFinance).

Cleans and scores the news for sentiment and emotions.

Merges news with stock data and prepares features for the model.

Makes a prediction for the next trading day and updates the output files.

Pushes the updated data to be displayed in the Streamlit dashboard.

You can also manually trigger the workflow to test or update the data pipeline.



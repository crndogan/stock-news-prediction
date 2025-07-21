import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Daily Stock News Dashboard", layout="wide")
st.title("Daily Stock News Dashboard")

#Load Data
@st.cache_data
def load_data():
    sentiment = pd.read_csv("sentiment_scores.csv", parse_dates=["date"] )
    predictions = pd.read_csv("BI_prediction_results.csv", parse_dates=["date"] )
    topics = pd.read_csv("daily_topic_distribution.csv", parse_dates=["date"] )
    wordfreq = pd.read_csv("topic_up_down.csv")
    return sentiment, predictions, topics, wordfreq

sent_df, pred_df, topics_df, wf_df = load_data()

# Sidebar: Date selector
max_date = sent_df["date"].max().date()
selected_date = st.sidebar.date_input("Select date", max_date, min_value=sent_df["date"].min().date(), max_value=max_date)

# --- Sentiment Trend ---
st.subheader("Sentiment Scores Over Time")
plot_df = sent_df.set_index("date")["sentiment_score"].loc[:pd.to_datetime(selected_date)]
st.line_chart(plot_df)

# --- Tomorrow's Prediction ---
st.subheader("Prediction for Next Trading Day")
today = pd.to_datetime(selected_date)
tomorrow_pred = pred_df[pred_df["date"] == today]
if not tomorrow_pred.empty:
    pred = tomorrow_pred.iloc[0]
    label = pred.get("predicted_label", "N/A")
    prob = pred.get("predicted_prob", None)
    if pd.notna(prob):
        st.metric(label="Change Prediction", value=label, delta=f"{prob:.2f}")
    else:
        st.metric(label="Change Prediction", value=label)
else:
    st.info("No prediction available for the selected date.")

# --- Topic Distribution ---
st.subheader(f"Topic Distribution on {selected_date}")
topic_row = topics_df[topics_df["date"] == pd.to_datetime(selected_date)]
if not topic_row.empty:
    weights = topic_row.drop(columns=["date", "dominant_topic"], errors="ignore").T
    weights.columns = ["weight"]
    fig, ax = plt.subplots()
    weights.plot(kind="bar", legend=False, ax=ax)
    ax.set_ylabel("Average Topic Weight")
    ax.set_xlabel("Topic")
    st.pyplot(fig)
else:
    st.info("No topic data for the selected date.")

# --- Word Clouds (Optional) ---
if st.checkbox("Show Word Clouds for Up/Down Headlines"):
    st.markdown("**Up Day Headlines**")
    up_wc = WordCloud(width=400, height=200).generate_from_frequencies(
        dict(wf_df[wf_df["market_label"]=="Up"]["count"].values)
    )
    st.image(up_wc.to_array(), use_column_width=False)
    st.markdown("**Down Day Headlines**")
    down_wc = WordCloud(width=400, height=200).generate_from_frequencies(
        dict(wf_df[wf_df["market_label"]=="Down"]["count"].values)
    )
    st.image(down_wc.to_array(), use_column_width=False)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("Data auto-updated via GitHub Actions")

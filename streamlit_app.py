import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Daily Stock News Dashboard", layout="wide")
st.title("📊 Daily Stock News Dashboard")

# --- Data Loader ---
@st.cache_data
def load_data():
    tone   = pd.read_excel("stock_news_tone.xlsx",  parse_dates=["date"])
    hist   = pd.read_csv(  "BI_prediction_results.csv", parse_dates=["date"])
    tomorrow = pd.read_csv("tomorrow_prediction.csv",   parse_dates=["date"])
    topics = pd.read_csv(  "topic_modeling_BI.csv",     parse_dates=["date"])
    wf     = pd.read_csv(  "topic_up_down.csv")  # labels = "Up"/"Down"
    return tone, hist, tomorrow, topics, wf

tone_df, hist_preds_df, tomorrow_df, topics_df, wf_df = load_data()

# --- Sidebar: Date Selector ---
max_date = tone_df["date"].max().date()
min_date = tone_df["date"].min().date()
selected_date = st.sidebar.date_input("Select date", max_date, min_value=min_date, max_value=max_date)

# --- 1) Price & Return Metrics ---
st.subheader(f"Price & Return on {selected_date}")
day = tone_df[tone_df["date"] == pd.to_datetime(selected_date)]
if not day.empty:
    close = day["close_price"].iloc[0]
    ret   = day["daily_return"].iloc[0]
    st.metric("Close Price", f"${close:,.2f}", delta=f"{ret:.2%}")
else:
    st.info("No price data for selected date.")

# --- 2) Sentiment Trend ---
st.subheader("Sentiment Compound Over Time")
sent_ts = tone_df.set_index("date")["sent_compound"]
st.line_chart(sent_ts)

# --- 3) Tomorrow's Prediction ---
st.subheader("Next Trading Day Prediction")
tom = tomorrow_df[tomorrow_df["date"] == pd.to_datetime(selected_date)]
if not tom.empty:
    mov  = tom["predicted_movement"].iloc[0]
    conf = tom["confidence"].iloc[0]
    st.metric("Predicted Movement", mov, f"{conf:.1%}")
else:
    st.info("No tomorrow-prediction available for that date.")

# --- 4) Historical Model Confidence ---
st.subheader("Historical Prediction Confidence")
hist_ts = hist_preds_df.set_index("date")["prediction_confidence"]
st.line_chart(hist_ts)

# --- 5) Topic of the Day ---
st.subheader("Topic Analysis")
tp = topics_df[topics_df["date"] == pd.to_datetime(selected_date)]
if not tp.empty:
    dom = tp["Dominant_Topic"].iloc[0]
    kws = tp["Topic_Keywords"].iloc[0]
    st.write(f"**Dominant Topic #{dom}:** {kws}")
    st.markdown("**Sample Headlines:**")
    st.write(tp["Headline"].tolist())
else:
    st.info("No topic data for that date.")

# --- 6) Word Clouds for Up/Down Headlines ---
if st.checkbox("Show Word Clouds (Up vs. Down Headlines)"):
    up_freq   = dict(wf_df[wf_df["label"]=="Up"]  [["word","count"]].values)
    down_freq = dict(wf_df[wf_df["label"]=="Down"][["word","count"]].values)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Up-Day Word Cloud**")
        wc1 = WordCloud(width=400, height=200).generate_from_frequencies(up_freq)
        st.image(wc1.to_array(), use_column_width=True)
    with col2:
        st.markdown("**Down-Day Word Cloud**")
        wc2 = WordCloud(width=400, height=200).generate_from_frequencies(down_freq)
        st.image(wc2.to_array(), use_column_width=True)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("Data automatically updated via GitHub Actions daily at 11 PM PT")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Daily Stock News Dashboard", layout="wide")
st.title("📊 Daily Stock News Dashboard")

# --- Data Loader ---
@st.cache_data
def load_data():
    base     = "notebooks"
    tone     = pd.read_excel(f"{base}/stock_news_tone.xlsx",         parse_dates=["date"])
    hist     = pd.read_csv(    f"{base}/BI_prediction_results.csv",  parse_dates=["date"])
    tomorrow = pd.read_csv(    f"{base}/tomorrow_prediction.csv",    parse_dates=["date"])
    topics   = pd.read_csv(    f"{base}/topic_modeling_BI.csv",      parse_dates=["date"])
    wf       = pd.read_csv(    f"{base}/topic_up_down.csv")          # labels = "Up"/"Down"
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
    st.metric("Close Price", f"${day['close_price'].iloc[0]:,.2f}", delta=f"{day['daily_return'].iloc[0]:.2%}")
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
    st.metric("Predicted Movement", tom["predicted_movement"].iloc[0], f"{tom['confidence'].iloc[0]:.1%}")
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
    st.write(f"**Dominant Topic #{tp['Dominant_Topic'].iloc[0]}:** {tp['Topic_Keywords'].iloc[0]}")
    st.markdown("**Sample Headlines:**")
    st.write(tp["Headline"].tolist())
else:
    st.info("No topic data for that date.")

# --- 6) Word Clouds for Up/Down Headlines ---
if st.checkbox("Show Word Clouds (Up vs. Down Headlines)"):
    up_freq   = dict(wf_df[wf_df["label"]=="Up"][["word","count"]].values)
    down_freq = dict(wf_df[wf_df["label"]=="Down"][["word","count"]].values)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Up-Day Word Cloud**")
        st.image(WordCloud(width=400, height=200).generate_from_frequencies(up_freq).to_array(), use_column_width=True)
    with col2:
        st.markdown("**Down-Day Word Cloud**")
        st.image(WordCloud(width=400, height=200).generate_from_frequencies(down_freq).to_array(), use_column_width=True)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("Data auto-updated via GitHub Actions daily at 11 PM PT")

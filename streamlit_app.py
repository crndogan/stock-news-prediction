import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from wordcloud import WordCloud

st.set_page_config(page_title="📊 News Headline Sentiment & Prediction Dashboard", layout="wide")
st.title(" Daily Stock Prediction & News Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    base     = "notebooks"
    tone     = pd.read_excel(f"{base}/stock_news_tone.xlsx",          parse_dates=["date"])
    hist     = pd.read_csv(f"{base}/BI_prediction_results.csv",       parse_dates=["date"])
    tomorrow = pd.read_csv(f"{base}/tomorrow_prediction.csv",         parse_dates=["date"])
    topics   = pd.read_csv(f"{base}/topic_modeling_BI.csv",           parse_dates=["date"])
    wf       = pd.read_csv(f"{base}/topic_up_down.csv")
    prices   = pd.read_csv(f"{base}/sp500_cleaned.csv",               parse_dates=["date"])
    return tone, hist, prices, tomorrow, topics, wf

tone_df, hist_df, sp500_df, tomorrow_df, topics_df, wf_df = load_data()

# --- Latest Available Date ---
today = tone_df["date"].max()
st.sidebar.info(f"Latest data: {today.date()}")

# --- Section 1: Prediction for Today ---
st.header("1. 📈 Next Trading Day Prediction")

prediction = tomorrow_df[tomorrow_df["date"] == today]
if not prediction.empty:
    st.metric("📊 Predicted Movement", prediction["predicted_movement"].iloc[0],
              f"{prediction['confidence'].iloc[0]:.1%}")
else:
    st.warning("Prediction for today is not yet available.")

# --- Section 2: Classification & Sentiment Summary ---
st.header("2. 📉 Classification & Sentiment Summary")

today_sent = tone_df[tone_df["date"] == today]
if not today_sent.empty:
    cols = st.columns(4)
    cols[0].metric("📍 Label", today_sent["label"].iloc[0])
    cols[1].metric("😐 Compound Sentiment", round(today_sent["sent_compound"].iloc[0], 3))
    cols[2].metric("😊 Positive", today_sent["emo_positive"].iloc[0])
    cols[3].metric("😠 Negative", today_sent["emo_negative"].iloc[0])
else:
    st.info("No sentiment data for today.")

# --- Section 3: Actual vs Predicted Prices ---
st.header("3. 📉 Actual vs Predicted Results")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(hist_df["date"], hist_df["actual_label"], color='green', label="Actual")
ax.plot(hist_df["date"], hist_df["predicted_label"], color='red', label="Predicted")
ax.set_title("Actual vs Predicted Labels")
ax.set_xlabel("Date")
ax.set_ylabel("Label (Up/Down)")
ax.legend()
st.pyplot(fig)

# --- Section 4: S&P 500 Data Table ---
st.header("4. 📋 S&P 500 Table")
sp_today = sp500_df[sp500_df["date"] == today]
if not sp_today.empty:
    st.dataframe(sp_today)
else:
    st.warning("No S&P 500 data for today.")

# --- Section 5: Topic of the Day ---
st.header("5. 🧠 Topic Modeling Results")
topic_today = topics_df[topics_df["date"] == today]
if not topic_today.empty:
    st.write(f"**Dominant Topic #{topic_today['Dominant_Topic'].iloc[0]}**")
    st.write(f"**Keywords:** {topic_today['Topic_Keywords'].iloc[0]}")
    st.markdown("**📰 Sample Headlines:**")
    for h in topic_today["Headline"].tolist():
        st.write(f"- {h}")
else:
    st.info("No topic results available for today.")

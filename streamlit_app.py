import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from wordcloud import WordCloud
import os
from pandas.tseries.offsets import BDay
import numpy as np

# PAGE SETUP
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

# CSS STYLING
st.markdown("""
    <style>
        .main {
            background-color: #eaf4fb;
        }
        .block-container {
            padding-top: 1rem;
        }
        .sidebar .sidebar-content {
            background-color: #d1ecf1;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Stock News & Market Movement Prediction")

# LOAD DATA
@st.cache_data(ttl=3600)
def load_data(version=None):
    base = "notebooks"
    tone = pd.read_excel(f"{base}/stock_news_tone.xlsx", parse_dates=["date"])
    hist = pd.read_csv(f"{base}/prediction_results.csv", parse_dates=["date"])
    prices = pd.read_csv(f"{base}/sp500_cleaned.csv", parse_dates=["date"])
    tomorrow = pd.read_csv(f"{base}/tomorrow_prediction.csv", parse_dates=["date"])
    topics = pd.read_csv(f"{base}/topic_modeling.csv", parse_dates=["date"])
    topic_change = pd.read_csv(f"{base}/topic_up_down.csv")
    return tone, hist, prices, tomorrow, topics, topic_change

def version_stamp():
    base = "notebooks"
    files = [
        f"{base}/stock_news_tone.xlsx", f"{base}/prediction_results.csv",
        f"{base}/sp500_cleaned.csv", f"{base}/tomorrow_prediction.csv",
        f"{base}/topic_modeling.csv", f"{base}/topic_up_down.csv"
    ]
    return tuple(os.path.getmtime(p) for p in files if os.path.exists(p))

tone_df, hist_df, sp500_df, tomorrow_df, topics_df, topic_change_df = load_data(version_stamp())

# DETERMINE TODAY
dfs = [tone_df, hist_df, sp500_df, tomorrow_df, topics_df]
today = max(df["date"].max() for df in dfs if not df.empty).normalize()
st.sidebar.info(f"Latest data: {today.date()}")

# SIDEBAR FILTERS
st.sidebar.markdown("### 🔍 Filters")
st.sidebar.markdown("Use the filters below to refine analysis")
selected_topic = st.sidebar.selectbox("Filter by Topic", options=["All"] + sorted(topics_df['Dominant_Topic'].unique()))
selected_sentiment = st.sidebar.slider("Filter by Compound Sentiment", min_value=-1.0, max_value=1.0, value=(-1.0, 1.0))

# NEXT TRADING DAY PREDICTION
st.header("Next Trading Day Prediction")
next_td = today + BDay(1)
pred_row = tomorrow_df[tomorrow_df["date"] == next_td]
if pred_row.empty and not tomorrow_df.empty:
    pred_row = tomorrow_df.loc[[tomorrow_df["date"].idxmax()]]
if not pred_row.empty:
    st.metric("Predicted Movement",
              pred_row["predicted_movement"].iloc[0],
              f"{pred_row['confidence'].iloc[0]:.1%}")
else:
    st.warning("No prediction available for today.")

# SENTIMENT SUMMARY
st.header("Classification & Sentiment Summary")
today_sent = tone_df[tone_df["date"] == today]
if not today_sent.empty:
    cols = st.columns(3)
    cols[0].metric("Compound Sentiment", round(today_sent["sent_compound"].iloc[0], 3))
    cols[1].metric("Positive", today_sent["emo_positive"].iloc[0])
    cols[2].metric("Negative", today_sent["emo_negative"].iloc[0])
else:
    st.info("No sentiment data available.")

# HISTORICAL PERFORMANCE (FILTERED)
st.header("Historical Prediction Performance (Filtered by Sentiment)")
selected_date = st.sidebar.date_input(
    "Select a date to view history up to",
    value=today,
    min_value=hist_df["date"].min().date(),
    max_value=hist_df["date"].max().date()
)

filtered_tone_df = tone_df[tone_df["sent_compound"].between(*selected_sentiment)]
filtered_hist = pd.merge(hist_df, filtered_tone_df[["date"]], on="date", how="inner")
filtered_hist = filtered_hist[filtered_hist["date"] <= pd.to_datetime(selected_date)].copy()

label_map = {"Up": 1, "Down": 0}
filtered_hist["actual_numeric"] = filtered_hist["actual_label"].map(label_map)
filtered_hist["predicted_numeric"] = filtered_hist["predicted_label"].map(label_map)

# ACTUAL vs PREDICTED SCATTER PLOT
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(filtered_hist["actual_numeric"], filtered_hist["predicted_numeric"], alpha=0.5, color="cornflowerblue", edgecolors="w", s=50)

if filtered_hist["actual_numeric"].nunique() > 1:
    z = np.polyfit(filtered_hist["actual_numeric"], filtered_hist["predicted_numeric"], 1)
    p = np.poly1d(z)
    ax2.plot(filtered_hist["actual_numeric"], p(filtered_hist["actual_numeric"]), color="blue", alpha=0.6, linewidth=2, label="Trend")

min_val = 0
max_val = 1
ax2.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', label='Perfect Prediction')

ax2.set_title("Actual vs Predicted Market Labels", fontsize=14)
ax2.set_xlabel("Actual Label (0 = Down, 1 = Up)")
ax2.set_ylabel("Predicted Label (0 = Down, 1 = Up)")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.set_xlim(-0.1, 1.1)
ax2.set_ylim(-0.1, 1.1)
ax2.legend()
st.pyplot(fig2)

# CLASSIFICATION METRICS
metrics_df = filtered_hist.dropna(subset=["actual_numeric", "predicted_numeric"])
y_true = metrics_df["actual_numeric"]
y_pred = metrics_df["predicted_numeric"]

if len(y_true) > 0 and y_true.nunique() == 2:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    st.subheader("Classification Metrics")
    st.markdown(f"""
    - **Accuracy:** {acc:.2%}  
    - **F1 Score:** {f1:.2f}  
    - **Precision:** {prec:.2f}  
    - **Recall:** {rec:.2f}
    """)
else:
    st.info("Not enough variation in filtered labels to compute metrics.")

# S&P 500 MARKET DATA (TABLE ONLY)
st.header("Market Close Data")
last_7_days = today - timedelta(days=7)
sp_week = sp500_df[sp500_df["date"] >= last_7_days].copy()

if not sp_week.empty:
    st.dataframe(sp_week.sort_values("date", ascending=False))
else:
    st.info("No S&P 500 data available for the past week.")

# TOPICS FROM LAST 7 DAYS
st.header("Topics from Last 7 Days")
topics_week = topics_df[topics_df["date"] >= last_7_days].sort_values("date", ascending=False)

if not topics_week.empty:
    for _, row in topics_week.iterrows():
        st.markdown(f"""
        <div style='padding: 8px; margin-bottom: 6px; background-color: #e9f5ff; border-left: 5px solid #007acc; border-radius: 4px;'>
            <b>{row['date'].date()} - Topic #{row['Dominant_Topic']}</b><br>
            {row['Topic_Keywords']}
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No topic modeling data available for the past 7 days.")

# TOPIC TRENDS WORDCLOUD
st.header("Topic Trends WordCloud")
if "word" in topic_change_df.columns and "label" in topic_change_df.columns:
    topic_change_df.dropna(subset=["word", "label"], inplace=True)
    text_up = " ".join(topic_change_df[topic_change_df["label"] == "Up"]["word"].astype(str))
    text_down = " ".join(topic_change_df[topic_change_df["label"] == "Down"]["word"].astype(str))
    wc_up = WordCloud(background_color='white', colormap='Greens').generate(text_up)
    wc_down = WordCloud(background_color='white', colormap='Reds').generate(text_down)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Topics Trending Up")
        fig_up, ax_up = plt.subplots(figsize=(6, 4))
        ax_up.imshow(wc_up, interpolation='bilinear')
        ax_up.axis("off")
        st.pyplot(fig_up)

    with col2:
        st.subheader("Topics Trending Down")
        fig_down, ax_down = plt.subplots(figsize=(6, 4))
        ax_down.imshow(wc_down, interpolation='bilinear')
        ax_down.axis("off")
        st.pyplot(fig_down)
else:
    st.warning("Topic change data is missing required columns.")



import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from datetime import timedelta
from pandas.tseries.offsets import BDay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from wordcloud import WordCloud
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

# Auto-refresh source: "remote" (GitHub raw URLs) or "local" (repo files)
DATA_MODE = "remote"  
RAW_BASE = st.secrets.get(
    "RAW_BASE",
    "https://raw.githubusercontent.com/crndogan/stock-news-prediction/main/notebooks"
)
LOCAL_BASE = "notebooks"

PRIMARY = "#0B6EFD"
BG_SOFT = "#F5F8FC"

st.markdown(f"""
<style>
  .main {{ background:{BG_SOFT}; }}
  .block-container {{ padding-top: .75rem; }}
  .kpi-card {{
    padding: 12px 14px; background: #fff; border: 1px solid #e9eef5;
    border-radius: 14px; box-shadow: 0 1px 2px rgba(0,0,0,.04);
  }}
  .section-title {{ margin:.5rem 0 .25rem; font-weight:700; font-size:1.1rem; color:#0f1a2a; }}
</style>
""", unsafe_allow_html=True)

st.title("Stock News & Market Movement Prediction")

# ---------------- HELPERS ----------------
def read_csv_remote(name):
    return pd.read_csv(f"{RAW_BASE}/{name}", parse_dates=["date"])

def read_excel_remote(name):
    return pd.read_excel(f"{RAW_BASE}/{name}", parse_dates=["date"])

def read_csv_local(name):
    return pd.read_csv(os.path.join(LOCAL_BASE, name), parse_dates=["date"])

def read_excel_local(name):
    return pd.read_excel(os.path.join(LOCAL_BASE, name), parse_dates=["date"])

# ---------------- DATA LOADING ----------------
@st.cache_data(ttl=900)  # refresh from remote every 15 minutes; click button below to force-clear
def load_data_remote():
    tone = read_excel_remote("stock_news_tone.xlsx")
    hist = read_csv_remote("prediction_results.csv")
    prices = read_csv_remote("sp500_cleaned.csv")
    tomorrow = read_csv_remote("tomorrow_prediction.csv")
    topics = read_csv_remote("topic_modeling.csv")
    topic_change = read_csv_remote("topic_up_down.csv")
    return tone, hist, prices, tomorrow, topics, topic_change

@st.cache_data(ttl=0)    # local: rely on manual cache clear or app redeploy
def load_data_local():
    tone = read_excel_local("stock_news_tone.xlsx")
    hist = read_csv_local("prediction_results.csv")
    prices = read_csv_local("sp500_cleaned.csv")
    tomorrow = read_csv_local("tomorrow_prediction.csv")
    topics = read_csv_local("topic_modeling.csv")
    topic_change = read_csv_local("topic_up_down.csv")
    return tone, hist, prices, tomorrow, topics, topic_change

# Refresh button
top_l, _ = st.columns([1,1])
with top_l:
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()

# Load (remote preferred)
if DATA_MODE == "remote":
    tone_df, hist_df, sp500_df, tomorrow_df, topics_df, topic_change_df = load_data_remote()
else:
    tone_df, hist_df, sp500_df, tomorrow_df, topics_df, topic_change_df = load_data_local()

# Normalize dates + topic dtype
for df_ in (tone_df, hist_df, sp500_df, tomorrow_df, topics_df):
    if not df_.empty:
        df_["date"] = pd.to_datetime(df_["date"]).dt.normalize()
if "Dominant_Topic" in topics_df.columns:
    topics_df["Dominant_Topic"] = topics_df["Dominant_Topic"].astype(str)

# ---------------- BASIC DATES ----------------
dfs = [tone_df, hist_df, sp500_df, tomorrow_df, topics_df]
today = max(df_["date"].max() for df_ in dfs if not df_.empty).normalize()
st.sidebar.info(f"📅 Latest data date: **{today.date()}**")

# ---------------- FILTERS ----------------
st.sidebar.markdown("### 🔍 Filters")
topic_options = ["All"]
if not topics_df.empty and "Dominant_Topic" in topics_df.columns:
    topic_options += sorted(topics_df["Dominant_Topic"].dropna().astype(str).unique().tolist())

selected_topic = st.sidebar.selectbox("Filter by Topic", options=topic_options, index=0)
selected_sentiment = st.sidebar.slider("Filter by Compound Sentiment", -1.0, 1.0, value=(-1.0, 1.0))

if not hist_df.empty:
    min_hist_date = hist_df["date"].min().date()
    max_hist_date = max(today.date(), hist_df["date"].max().date())
    selected_date = st.sidebar.date_input(
        "Show history up to",
        value=max_hist_date,
        min_value=min_hist_date,
        max_value=max_hist_date
    )
else:
    selected_date = today.date()

# ---------------- NEXT DAY PREDICTION ----------------
st.markdown('<div class="section-title">Next Trading Day Prediction</div>', unsafe_allow_html=True)
next_td = today + BDay(1)
pred_row = tomorrow_df[tomorrow_df["date"] == next_td]
if pred_row.empty and not tomorrow_df.empty:
    pred_row = tomorrow_df.loc[[tomorrow_df["date"].idxmax()]]

if not pred_row.empty:
    c1, c2 = st.columns(2)
    c1.metric("Predicted Movement", pred_row["predicted_movement"].iloc[0])
    c2.metric("Confidence", f"{pred_row['confidence'].iloc[0]:.1%}")
else:
    st.warning("No prediction available for the next trading day.")

# ---------------- TODAY SENTIMENT ----------------
st.markdown('<div class="section-title">Today’s Sentiment Snapshot</div>', unsafe_allow_html=True)
today_sent = tone_df[tone_df["date"] == today]
if not today_sent.empty:
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='kpi-card'><b>Compound</b><br><span style='font-size:1.4rem;'>{today_sent['sent_compound'].iloc[0]:.3f}</span></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card'><b>Emotion: Positive</b><br><span style='font-size:1.4rem;'>{today_sent['emo_positive'].iloc[0]:.3f}</span></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card'><b>Emotion: Negative</b><br><span style='font-size:1.4rem;'>{today_sent['emo_negative'].iloc[0]:.3f}</span></div>", unsafe_allow_html=True)
else:
    st.info("No sentiment data for today.")

# ---------------- FILTERED HISTORY BUILD ----------------
tone_dates = tone_df.loc[tone_df["sent_compound"].between(*selected_sentiment), ["date"]].drop_duplicates()

if selected_topic != "All" and not topics_df.empty:
    topic_dates = topics_df.loc[topics_df["Dominant_Topic"] == str(selected_topic), ["date"]].drop_duplicates()
    driver_dates = pd.merge(tone_dates, topic_dates, on="date", how="inner")
else:
    driver_dates = tone_dates

driver_dates = driver_dates[driver_dates["date"] <= pd.to_datetime(selected_date)]

filtered_hist = (
    pd.merge(hist_df.copy(), driver_dates, on="date", how="inner")
      .sort_values("date")
      .reset_index(drop=True)
)

label_map = {"Up": 1, "Down": 0, 1: 1, 0: 0, "1": 1, "0": 0}
for col in ["actual_label", "predicted_label"]:
    if col in filtered_hist.columns:
        filtered_hist[col] = filtered_hist[col].astype(str).str.strip()
filtered_hist["actual_numeric"] = filtered_hist["actual_label"].map(label_map)
filtered_hist["predicted_numeric"] = filtered_hist["predicted_label"].map(label_map)

# ---------------- CHART ----------------
st.markdown('<div class="section-title">Actual vs Predicted Market Direction</div>', unsafe_allow_html=True)
if not filtered_hist.empty:
    chart_df = filtered_hist[["date", "actual_numeric", "predicted_numeric"]].melt(
        id_vars="date", var_name="Series", value_name="Value"
    ).replace({"actual_numeric": "Actual", "predicted_numeric": "Predicted"})

    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Value:Q", title="Direction (0=Down, 1=Up)",
                    scale=alt.Scale(domain=[-0.05, 1.05])),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("date:T", title="Date"), "Series:N", alt.Tooltip("Value:Q", title="Direction")]
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No rows match the current filters (topic, sentiment, date).")

# ---------------- METRICS (numbers only) ----------------
st.markdown('<div class="section-title">Classification Metrics</div>', unsafe_allow_html=True)
metrics_df = filtered_hist.dropna(subset=["actual_numeric", "predicted_numeric"])
if not metrics_df.empty and metrics_df["actual_numeric"].nunique() == 2:
    y_true = metrics_df["actual_numeric"].astype(int)
    y_pred = metrics_df["predicted_numeric"].astype(int)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{acc:.2%}")
    k2.metric("F1 Score", f"{f1:.2f}")
    k3.metric("Precision", f"{prec:.2f}")
    k4.metric("Recall", f"{rec:.2f}")
    st.caption(f"Based on {len(metrics_df)} trading days under current filters. Source: {DATA_MODE.upper()}")
else:
    st.info("Not enough class variation under current filters to compute metrics (need both Up and Down).")

# ---------------- S&P TABLE (colored Direction) ----------------
st.markdown('<div class="section-title">Recent S&P 500 Market Close</div>', unsafe_allow_html=True)
last_7_days = today - timedelta(days=7)
sp_week = sp500_df[sp500_df["date"] >= last_7_days].copy().sort_values("date", ascending=False)

if not sp_week.empty:
    if "close_price" in sp_week.columns:
        sp_week["prev_close"] = sp_week["close_price"].shift(-1)
        sp_week["Return"] = (sp_week["close_price"] / sp_week["prev_close"] - 1.0) * 100
        sp_week["Direction"] = np.where(sp_week["Return"] >= 0, "Up", "Down")
        sp_week["Move"] = sp_week["Return"].map(lambda x: ("" if pd.isna(x) else ("↑" if x >= 0 else "↓")) + ("" if pd.isna(x) else f" {x:,.2f}%"))
        view = sp_week[["date", "close_price", "Move", "Direction"]].rename(columns={"date":"Date","close_price":"Close"})

        def color_dir_col(s: pd.Series):
            # apply to the whole column at once
            return np.where(s.eq("Up"), "background-color: #E6F4EA; color:#0f5132;",
                            np.where(s.eq("Down"), "background-color: #FDECEC; color:#842029;", ""))

        styled = (view.style
                  .format({"Close": "{:,.2f}"})
                  .apply(lambda df: [""]*len(df), axis=1)  # keep other cols default
                  .apply(color_dir_col, subset=["Direction"]))
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(sp_week, use_container_width=True, hide_index=True)
else:
    st.info("No S&P 500 data in the last 7 days.")

# ---------------- TOPICS (last 7 days) ----------------
st.markdown('<div class="section-title">Topics from the Last 7 Days</div>', unsafe_allow_html=True)
topics_week = topics_df[topics_df["date"] >= last_7_days].sort_values("date", ascending=False)
if not topics_week.empty and {"Dominant_Topic","Topic_Keywords"}.issubset(topics_week.columns):
    for _, row in topics_week.iterrows():
        st.markdown(f"""
        <div style='padding:10px;margin-bottom:8px;background:#fff;border:1px solid #e9eef5;border-radius:12px;'>
            <b>{row['date'].date()}</b> — <span style="color:{PRIMARY}">Topic #{row['Dominant_Topic']}</span><br>
            <span style="opacity:.9">{row['Topic_Keywords']}</span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No topic modeling data available for the past 7 days.")

# ---------------- WORDCLOUDS ----------------
st.markdown('<div class="section-title">Topic Trends WordCloud</div>', unsafe_allow_html=True)
if {"word", "label"}.issubset(set(topic_change_df.columns)):
    topic_change_df = topic_change_df.dropna(subset=["word", "label"])
    text_up = " ".join(topic_change_df[topic_change_df["label"].astype(str).str.lower() == "up"]["word"].astype(str))
    text_down = " ".join(topic_change_df[topic_change_df["label"].astype(str).str.lower() == "down"]["word"].astype(str))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trending on Up Days")
        fig_up, ax_up = plt.subplots(figsize=(6, 4))
        wc_up = WordCloud(background_color='white', colormap='Greens').generate(text_up if text_up else "NoData")
        ax_up.imshow(wc_up, interpolation='bilinear'); ax_up.axis("off")
        st.pyplot(fig_up)
    with col2:
        st.subheader("Trending on Down Days")
        fig_down, ax_down = plt.subplots(figsize=(6, 4))
        wc_down = WordCloud(background_color='white', colormap='Reds').generate(text_down if text_down else "NoData")
        ax_down.imshow(wc_down, interpolation='bilinear'); ax_down.axis("off")
        st.pyplot(fig_down)
else:
    st.warning("Topic change data is missing required columns: 'word' and 'label'.")

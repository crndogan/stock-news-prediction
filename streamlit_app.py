import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from wordcloud import WordCloud

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #eaf4fb;
        }
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Stock News & Market Movement Prediction")

# --- Load Data ---
@st.cache_data
def load_data():
    base = "notebooks"
    tone = pd.read_excel(f"{base}/stock_news_tone.xlsx", parse_dates=["date"])
    hist = pd.read_csv(f"{base}/prediction_results.csv", parse_dates=["date"])
    prices = pd.read_csv(f"{base}/sp500_cleaned.csv", parse_dates=["date"])
    tomorrow = pd.read_csv(f"{base}/tomorrow_prediction.csv", parse_dates=["date"])
    topics = pd.read_csv(f"{base}/topic_modeling.csv", parse_dates=["date"])
    topic_change = pd.read_csv(f"{base}/topic_up_down.csv")
    return tone, hist, prices, tomorrow, topics, topic_change

tone_df, hist_df, sp500_df, tomorrow_df, topics_df, topic_change_df = load_data()

# --- Use max valid sentiment date ---
valid_sentiment = tone_df[tone_df["emo_positive"] > 0]
today = valid_sentiment["date"].max()
st.sidebar.info(f"Latest data: {today.date()}")

st.sidebar.markdown("### 💡 Daily Tip")
st.sidebar.success("Use the date filter below to view model behavior on past days!")

# --- Prediction ---
st.header("1. Next Trading Day Prediction")
pred_row = tomorrow_df[tomorrow_df["date"] == today]
if not pred_row.empty:
    st.metric("Predicted Movement", pred_row["predicted_movement"].iloc[0],
              f"{pred_row['confidence'].iloc[0]:.1%}")
else:
    st.warning("No prediction available for today.")

# --- Sentiment Summary ---
st.header("2. Classification & Sentiment Summary")
today_sent = tone_df[tone_df["date"] == today]
if not today_sent.empty:
    cols = st.columns(4)
    cols[0].metric("Label", today_sent["label"].iloc[0])
    cols[1].metric("Compound Sentiment", round(today_sent["sent_compound"].iloc[0], 3))
    cols[2].metric("Positive", today_sent["emo_positive"].iloc[0])
    cols[3].metric("Negative", today_sent["emo_negative"].iloc[0])
else:
    st.info("No sentiment data available.")

# --- Historical Classification Chart & Metrics ---
st.header("3. Historical Prediction Performance")
selected_date = st.sidebar.date_input("Select a date to view history up to", value=today,
                                      min_value=hist_df["date"].min().date(),
                                      max_value=hist_df["date"].max().date())
filtered_hist = hist_df[hist_df["date"] <= pd.to_datetime(selected_date)]
label_map = {"Up": 1, "Down": 0}
filtered_hist["actual_numeric"] = filtered_hist["actual_label"].map(label_map)
filtered_hist["predicted_numeric"] = filtered_hist["predicted_label"].map(label_map)

fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(filtered_hist["date"], filtered_hist["actual_numeric"], label="Actual", color="green", marker='o')
ax.scatter(filtered_hist["date"], filtered_hist["predicted_numeric"], label="Predicted", color="red", marker='x')
ax.set_title("Actual vs Predicted (Up = 1, Down = 0)")
ax.set_ylabel("Label")
ax.set_xlabel("Date")
ax.legend()
st.pyplot(fig)

# --- Classification Metrics ---
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
    st.info("Not enough class variation or valid data to compute metrics.")

# --- S&P 500 Daily Table & Price Trend ---
st.header("4. Market Close Data")
sp_today = sp500_df[sp500_df["date"] == today]
if not sp_today.empty:
    st.dataframe(sp_today)

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(sp500_df["date"], sp500_df["close"], color="blue")
    ax2.set_title("S&P 500 Closing Price Trend")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Close Price")
    st.pyplot(fig2)
else:
    st.info("No S&P data available for today.")

# --- Topic Modeling ---
st.header("5. Topic of the Day")
topic_today = topics_df[topics_df["date"] == today]
if not topic_today.empty:
    st.markdown(f"""
    <div style='padding: 10px; background-color: #cce5ff; border-radius: 6px; width:fit-content;'>
        <b>Topic #{topic_today['Dominant_Topic'].iloc[0]}</b>: {topic_today['Topic_Keywords'].iloc[0]}
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("No topic modeling data available for today.")

# --- WordCloud of Up & Down Topics ---
st.header("6. Topic Trends WordCloud")
topic_change_df.dropna(subset=["Topic", "Direction"], inplace=True)
text_up = " ".join(topic_change_df[topic_change_df["Direction"] == "Up"]["Topic"].astype(str))
text_down = " ".join(topic_change_df[topic_change_df["Direction"] == "Down"]["Topic"].astype(str))

wc_up = WordCloud(background_color='white', colormap='Greens').generate(text_up)
wc_down = WordCloud(background_color='white', colormap='Reds').generate(text_down)

st.subheader("Topics Trending Up")
fig_up, ax_up = plt.subplots()
ax_up.imshow(wc_up, interpolation='bilinear')
ax_up.axis("off")
st.pyplot(fig_up)

st.subheader("Topics Trending Down")
fig_down, ax_down = plt.subplots()
ax_down.imshow(wc_down, interpolation='bilinear')
ax_down.axis("off")
st.pyplot(fig_down)

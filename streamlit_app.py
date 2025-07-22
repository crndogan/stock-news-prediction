import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
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
    return tone, hist, prices, tomorrow, topics

tone_df, hist_df, sp500_df, tomorrow_df, topics_df = load_data()

# --- Use max valid sentiment date ---
valid_sentiment = tone_df[tone_df["emo_positive"] > 0]
today = valid_sentiment["date"].max()
st.sidebar.info(f"Latest data: {today.date()}")

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

    fig_cm, ax_cm = plt.subplots(figsize=(2.5, 2.5))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["Down", "Up"])
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("Confusion Matrix", fontsize=10)
    st.pyplot(fig_cm)
else:
    st.info("Not enough class variation or valid data to compute metrics.")

# --- S&P 500 Daily Table ---
st.header("4. Market Close Data")
sp_today = sp500_df[sp500_df["date"] == today]
if not sp_today.empty:
    st.dataframe(sp_today)
else:
    st.info("No S&P data available for today.")

# --- Topic Modeling ---
st.header("5. Topic of the Day")
topic_today = topics_df[topics_df["date"] == today]
if not topic_today.empty:
    st.write(f"**Topic #{topic_today['Dominant_Topic'].iloc[0]}**: {topic_today['Topic_Keywords'].iloc[0]}")
    st.markdown("**Sample Headlines:**")
    for h in topic_today["Headline"].tolist():
        st.write(f"- {h}")
else:
    st.info("No topic modeling data available for today.")

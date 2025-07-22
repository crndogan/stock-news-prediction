import streamlit as st 
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Daily Stock News Dashboard", layout="wide")
st.title("📊 Daily Stock News Dashboard")

# ── Data Loader ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    base     = "notebooks"
    tone     = pd.read_excel(f"{base}/stock_news_tone.xlsx",          parse_dates=["date"])
    hist     = pd.read_csv(f"{base}/BI_prediction_results.csv",       parse_dates=["date"])
    tomorrow = pd.read_csv(f"{base}/tomorrow_prediction.csv",         parse_dates=["date"])
    topics   = pd.read_csv(f"{base}/topic_modeling_BI.csv",           parse_dates=["date"])
    wf       = pd.read_csv(f"{base}/topic_up_down.csv")
    prices   = pd.read_csv(f"{base}/sp500_prices_master.csv",         parse_dates=["date"])
    return tone, hist, tomorrow, topics, wf, prices

tone_df, hist_df, tomorrow_df, topics_df, wf_df, prices_df = load_data()

# ── 1) Classification Performance Metrics ───────────────────────────────────
valid = hist_df.dropna(subset=["actual_label", "predicted_label"])
y_true, y_pred = valid["actual_label"], valid["predicted_label"]

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, pos_label="Up")
rec  = recall_score(y_true, y_pred, pos_label="Up")
f1   = f1_score(y_true, y_pred, pos_label="Up")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy",  f"{acc:.3f}")
c2.metric("Precision", f"{prec:.3f}")
c3.metric("Recall",    f"{rec:.3f}")
c4.metric("F1 Score",  f"{f1:.3f}")

st.markdown("---")

# ── 2) Monthly Accuracy Bar Chart ────────────────────────────────────────────
valid["month"] = valid["date"].dt.to_period("M").dt.to_timestamp()
monthly = (
    valid
    .groupby("month")
    .apply(lambda d: accuracy_score(d["actual_label"], d["predicted_label"]))
    .reset_index(name="accuracy")
)
st.subheader("Monthly Accuracy")
fig_m = px.bar(
    monthly, x="month", y="accuracy",
    labels={"month":"Month","accuracy":"Accuracy"},
    template="plotly_white", color="accuracy", color_continuous_scale="Viridis"
)
st.plotly_chart(fig_m, use_container_width=True)

st.markdown("---")

# ── 3) Sidebar: Date Selector ────────────────────────────────────────────────
all_dates = pd.concat([tone_df["date"], tomorrow_df["date"]]).drop_duplicates()
selected  = st.sidebar.date_input(
    "Select date",
    value=all_dates.max().date(),
    min_value=all_dates.min().date(),
    max_value=all_dates.max().date()
)
is_trading = selected in tone_df["date"].dt.date.values
if not is_trading:
    st.warning(f"⚠️ Market Closed on {selected}; showing next-day prediction only")
st.sidebar.markdown("---")
st.sidebar.write("Data auto-updated via GitHub Actions at 11 PM PT")

# ── 4) Top Row: Price, Prediction & Topic ────────────────────────────────────
r1, r2, r3 = st.columns(3)
day = tone_df[tone_df["date"] == pd.to_datetime(selected)]

if is_trading and not day.empty:
    price = day["close_price"].iloc[0]
    rtn   = day["daily_return"].iloc[0]
    dc    = "normal" if rtn >= 0 else "inverse"
    r1.metric("Close Price", f"${price:,.2f}", f"{rtn:.2%}", delta_color=dc)
else:
    r1.info("No price data")

tom = tomorrow_df[tomorrow_df["date"] == pd.to_datetime(selected)]
if not tom.empty:
    mv, cf = tom["predicted_movement"].iloc[0], tom["confidence"].iloc[0]
    r2.metric("Next-Day Movement", mv, f"{cf:.1%}")
else:
    r2.info("No prediction")

tp = topics_df[topics_df["date"] == pd.to_datetime(selected)]
if not tp.empty:
    dom, kws = tp["Dominant_Topic"].iloc[0], tp["Topic_Keywords"].iloc[0]
    r3.markdown(f"**Topic #{dom}:** {kws}")
else:
    r3.info("No topic data")

st.markdown("---")

# ── 5) Sentiment & Confidence ────────────────────────────────────────────────
cA, cB = st.columns((2,1))
with cA:
    st.subheader("Sentiment Compound Over Time")
    fig_s = px.line(
        tone_df, x="date", y="sent_compound",
        labels={"date":"","sent_compound":"Compound"},
        template="plotly_white"
    ).update_traces(line_color="royalblue")
    st.plotly_chart(fig_s, use_container_width=True)

with cB:
    st.subheader("Historical Prediction Confidence")
    fig_c = px.line(
        hist_df, x="date", y="prediction_confidence",
        labels={"date":"","prediction_confidence":"Confidence"},
        template="plotly_white"
    ).update_traces(line_color="darkorange")
    st.plotly_chart(fig_c, use_container_width=True)

st.markdown("---")

# ── 6) Topic Weights ─────────────────────────────────────────────────────────
st.subheader("Topic Weights on Selected Date")
bar = topics_df[topics_df["date"] == pd.to_datetime(selected)]
if not bar.empty:
    wts = bar.drop(columns=["date","Dominant_Topic","Topic_Keywords","Headline"], errors="ignore").T
    wts.columns = ["weight"]
    fig_w = px.bar(
        wts.reset_index(), x="index", y="weight",
        labels={"index":"Topic","weight":"Weight"},
        color="weight", color_continuous_scale="Blues",
        template="plotly_white"
    )
    st.plotly_chart(fig_w, use_container_width=True)
else:
    st.info("No topic weights")

# ── 7) Word Clouds ───────────────────────────────────────────────────────────
if st.checkbox("Show Word Clouds"):
    wc1, wc2 = st.columns(2)
    up   = dict(wf_df.query("market_label=='Up'")[["word","count"]].values)
    down = dict(wf_df.query("market_label=='Down'")[["word","count"]].values)
    with wc1:
        st.subheader("Up-Day Word Cloud")
        img = WordCloud(width=300, height=200, background_color="white").generate_from_frequencies(up)
        st.image(img.to_array(), use_column_width=True)
    with wc2:
        st.subheader("Down-Day Word Cloud")
        img = WordCloud(width=300, height=200, background_color="white").generate_from_frequencies(down)
        st.image(img.to_array(), use_container_width=True)

# ── 8) Actual vs Predicted Prices (SVR) ──────────────────────────────────────
st.markdown("---")
st.subheader("📉 Actual vs Predicted Prices (SVR)")

fig_ap = px.line(
    prices_df,
    x="date",
    y=["actual", "predicted"],
    labels={"value": "Closing Price (USD)", "variable": "Legend"},
    title="Actual vs Predicted Closing Prices",
    template="plotly_white"
)
fig_ap.update_traces(mode="lines+markers")
st.plotly_chart(fig_ap, use_container_width=True)

# ── 9) Monthly Closing Price ─────────────────────────────────────────────────
st.subheader("📈 SP500 Monthly Closing Trend")

monthly = (
    tone_df
    .set_index("date")["close_price"]
    .resample("M")
    .last()
    .reset_index()
)

fig_month = px.line(
    monthly,
    x="date",
    y="close_price",
    labels={"date": "Date", "close_price": "Closing Price (USD)"},
    template="plotly_white",
    title="SP500 Monthly Closing Price"
)
fig_month.update_traces(line_color="blue", mode="lines+markers")
st.plotly_chart(fig_month, use_container_width=True)

# ── 10) Stock Data Table ─────────────────────────────────────────────────────
st.subheader("📋 Stock Data Table (SVR File)")

formatted = prices_df.copy()
for col in ["actual", "predicted", "open", "high", "adj_close"]:
    formatted[col] = formatted[col].map("${:,.2f}".format)
formatted["volume"] = formatted["volume"].map("{:,}".format)

st.dataframe(
    formatted[["date", "open", "high", "actual", "predicted", "adj_close", "volume"]]
    .sort_values("date", ascending=False)
    .reset_index(drop=True),
    use_container_width=True
)

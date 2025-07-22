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
    tone     = pd.read_excel("stock_news_tone.xlsx",          parse_dates=["date"])
    hist     = pd.read_csv("BI_prediction_results.csv",       parse_dates=["date"])
    tomorrow = pd.read_csv("tomorrow_prediction.csv",         parse_dates=["date"])
    topics   = pd.read_csv("topic_modeling_BI.csv",           parse_dates=["date"])
    wf       = pd.read_csv("topic_up_down.csv")
    return tone, hist, tomorrow, topics, wf

tone_df, hist_df, tomorrow_df, topics_df, wf_df = load_data()

# ── 0) Performance Metrics ───────────────────────────────────────────────────
with st.expander("Performance Metrics ▶"):
    # compute global classification metrics
    y_true = hist_df["actual_label"].dropna()
    y_pred = hist_df.loc[y_true.index, "predicted_label"]
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label="Up"),
        "Recall":    recall_score(y_true, y_pred, pos_label="Up"),
        "F1 Score":  f1_score(y_true, y_pred, pos_label="Up"),
    }
    # display
    cols = st.columns(len(metrics))
    for (name, val), col in zip(metrics.items(), cols):
        col.metric(name, f"{val:.3f}")

    # deep dive: monthly accuracy
    hist_df["month"] = hist_df["date"].dt.to_period("M").dt.to_timestamp()
    monthly_acc = (
        hist_df
        .groupby("month")
        .apply(lambda d: accuracy_score(d["actual_label"], d["predicted_label"]))
        .reset_index(name="accuracy")
    )
    fig_monthly = px.bar(
        monthly_acc, x="month", y="accuracy",
        labels={"month":"Month","accuracy":"Accuracy"},
        title="Monthly Accuracy",
        template="plotly_white",
        color="accuracy", color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

st.markdown("---")

# ── Sidebar: Date Selector (includes future) ─────────────────────────────────
all_dates = pd.concat([tone_df["date"], tomorrow_df["date"]]).drop_duplicates()
selected = st.sidebar.date_input(
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

# ── Top Row: Price, Prediction & Topic ───────────────────────────────────────
c1, c2, c3 = st.columns(3)
day = tone_df[tone_df["date"] == pd.to_datetime(selected)]

# 1) Close Price & Return
if is_trading and not day.empty:
    close = day["close_price"].iloc[0]
    rtn   = day["daily_return"].iloc[0]
    delta_color = "normal" if rtn>=0 else "inverse"
    c1.metric("Close Price", f"${close:,.2f}", f"{rtn:.2%}", delta_color=delta_color)
else:
    c1.info("No price data")

# 2) Next-Day Prediction
tom = tomorrow_df[tomorrow_df["date"] == pd.to_datetime(selected)]
if not tom.empty:
    move = tom["predicted_movement"].iloc[0]
    conf = tom["confidence"].iloc[0]
    c2.metric("Next-Day Prediction", move, f"{conf:.1%}")
else:
    c2.info("No prediction")

# 3) Dominant Topic & Keywords
tp = topics_df[topics_df["date"] == pd.to_datetime(selected)]
if not tp.empty:
    dom = tp["Dominant_Topic"].iloc[0]
    kws = tp["Topic_Keywords"].iloc[0]
    c3.markdown(f"**Topic #{dom}:** {kws}")
else:
    c3.info("No topic data")

st.markdown("---")

# ── Trends: Sentiment & Model Confidence ──────────────────────────────────────
t1, t2 = st.columns((2,1))
with t1:
    st.subheader("Sentiment Compound Over Time")
    fig_sent = px.line(tone_df, x="date", y="sent_compound",
                       labels={"date":"","sent_compound":"Compound"},
                       template="plotly_white").update_traces(line_color="royalblue")
    st.plotly_chart(fig_sent, use_container_width=True)
with t2:
    st.subheader("Historical Prediction Confidence")
    fig_conf = px.line(hist_df, x="date", y="prediction_confidence",
                       labels={"date":"","prediction_confidence":"Confidence"},
                       template="plotly_white").update_traces(line_color="darkorange")
    st.plotly_chart(fig_conf, use_container_width=True)

st.markdown("---")

# ── Topic Weights Bar ────────────────────────────────────────────────────────
st.subheader("Topic Weights on Selected Date")
bar = topics_df[topics_df["date"] == pd.to_datetime(selected)]
if not bar.empty:
    weights = bar.drop(columns=["date","Dominant_Topic","Topic_Keywords","Headline"], errors="ignore").T
    weights.columns = ["weight"]
    fig_bar = px.bar(weights.reset_index(), x="index", y="weight",
                     labels={"index":"Topic","weight":"Weight"},
                     color="weight", color_continuous_scale="Blues",
                     template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No topic weights")

# ── Word Clouds (optional) ───────────────────────────────────────────────────
if st.checkbox("Show Word Clouds"):
    wc1, wc2 = st.columns(2)
    up   = dict(wf_df.query("market_label=='Up'")[["word","count"]].values)
    down = dict(wf_df.query("market_label=='Down'")[["word","count"]].values)
    with wc1:
        st.subheader("Up-Day Word Cloud")
        st.image(WordCloud(300,200,background_color="white").generate_from_frequencies(up).to_array())
    with wc2:
        st.subheader("Down-Day Word Cloud")
        st.image(WordCloud(300,200,background_color="white").generate_from_frequencies(down).to_array())


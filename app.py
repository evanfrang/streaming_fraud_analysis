import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ------------------------
# Page Layout
# ------------------------

st.set_page_config(
    page_title="Suspicious User Detection",
    page_icon="🔎",
    layout="wide"
)

st.title("Fradulent Streaming Data")

# ------------------------
# Data
# ------------------------

@st.cache_data
def load_user_activity():
    return pd.read_pickle("data/user_activity.pkl")

@st.cache_data
def load_user_days():
    return pd.read_pickle("data/user_day_counts.pkl")

user_activity = load_user_activity()
user_days = load_user_days()

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.title("Suspicious User Detection")

numeric_cols = user_activity.select_dtypes(include="number").columns.tolist()

selected_metric = st.sidebar.selectbox(
    "Filter metric", 
    numeric_cols, 
    index=numeric_cols.index("high_stream_days") if "total_streams" in numeric_cols else 0
)

threshold = st.sidebar.number_input(f"Minimum {selected_metric}", value=1.0)

sus_users = user_activity[user_activity[selected_metric] >= threshold]

st.write(f"### {len(sus_users)} Suspicious Users (filtered by {selected_metric} ≥ {round(threshold,3)})")

# -------------------------
# Leaderboard
# -------------------------
styled = sus_users.sort_values(selected_metric, ascending=False).set_index("user_id").style.format(
    precision=2
).background_gradient(cmap="Blues", subset=[selected_metric])

st.dataframe(styled, use_container_width=True, height=500)

# -------------------------
# Distribution plot
# -------------------------
c1, c2 = st.columns(2)

with c1:
    fig_all = px.histogram(user_activity, x=selected_metric, nbins=50,
                           title=f"Distribution of {selected_metric} (All Users)")
    st.plotly_chart(fig_all, use_container_width=True)

with c2:
    fig_sus = px.histogram(sus_users, x=selected_metric, nbins=50,
                           title=f"Distribution of {selected_metric} (Suspicious Only)")
    st.plotly_chart(fig_sus, use_container_width=True)

# -------------------------
# Per-user drill down
# -------------------------
st.write("## Explore a user")
user_choice = st.selectbox("Pick a user_id", sus_users["user_id"].unique())

if user_choice:
    try:
        # Daily Streams
        user_days = user_days[user_days["user_id"] == user_choice]
        fig2 = px.line(user_days, x="timestamp", y="daily_streams", 
                       title=f"Daily streams for User {user_choice}")
        #fig2.update_traces(line=dict(color="crimson", width=2))

        # Listening %
        fig3 = px.line(user_days, x="timestamp", y="listen_pct", title=f"Daily Listening % for User {user_choice}")
        #fig3.update_traces(line=dict(color="green", width=2))


        # Autocorrelation
        user_series = (
            user_days[user_days['user_id'] == user_choice]
            .set_index('timestamp')
            .asfreq('D', fill_value=0)['daily_streams']
        )
        ac = np.correlate(user_series - user_series.mean(),
                      user_series - user_series.mean(), mode='full')
        ac = ac[ac.size // 2:]  
        ac /= ac[0]
        lags = np.arange(1, len(ac))  
        ac = ac[1:]
        fig4 = px.line(
            x=lags,
            y=ac,
            labels={'x': 'Lag (days)', 'y': 'Autocorrelation'},
            title=f"Autocorrelation for User {user_choice}"
        )
        #fig4.update_traces(line=dict(color="purple", width=2))

        # Fourier Transform
        values = user_series.values - user_series.mean()
        n = len(values)

        fft_vals = np.fft.fft(values)
        freqs = np.fft.fftfreq(n, d=1)

        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        power = np.abs(fft_vals[pos_mask])**2
        fig5 = px.line(
            x=freqs,
            y=power,
            labels={'x': 'Frequency (cycles per day)', 'y': 'Power'},
            title=f"Fourier Transform Spectrum for User {user_choice}"
        )
        #fig5.update_traces(line=dict(color="orange", width=2))

        c1, c2 = st.columns(2)
        c1.plotly_chart(fig2, use_container_width=True)
        c2.plotly_chart(fig3, use_container_width=True)

        c3, c4 = st.columns(2)
        c3.plotly_chart(fig4, use_container_width=True)
        c4.plotly_chart(fig5, use_container_width=True)

    except FileNotFoundError:
        st.info("Per-day data not available — only summary features shown.")
        st.write(user_activity[user_activity["user_id"] == user_choice].T)


st.write("### Users flagged by Isolation Forest")

features = user_activity.select_dtypes(include="number").columns.tolist()
X = user_activity[features].drop(columns=["user_id"])

iso = IsolationForest(n_estimators=300, contamination=0.08, random_state=13)
iso.fit(X)

user_activity["anomaly_score"] = iso.decision_function(X)
user_activity["is_anomaly"] = iso.predict(X)
user_activity["is_bot"] = user_activity['user_id'] >= 301 
# the bots are 301 and up in this dataset
# this should be done better so we don't have issues later


ml_anomalies = user_activity[user_activity["is_anomaly"] == -1].copy()

st.dataframe(
    ml_anomalies.sort_values("anomaly_score").set_index("user_id"),
    use_container_width=True,
    height=500
)

user_activity["is_anomaly"] = user_activity["is_anomaly"].apply(lambda x: 1 if x==-1 else 0)
user_activity["is_bot"] = user_activity["is_bot"].astype(int)

#corrs = X.corrwith(user_activity["anomaly_score"]).sort_values(key=abs, ascending=False)
#st.bar_chart(corrs)

st.markdown("#### Isolation Forest Performance")

prec, rec, f1, _ = precision_recall_fscore_support(
    user_activity["is_bot"], user_activity["is_anomaly"], average=None
)

metrics_df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1 Score"],
    "Value": [prec, rec, f1]
})

scores_df = pd.DataFrame({
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1
}, index=["Normal", "Anomaly"])

st.table(scores_df)
cm = confusion_matrix(user_activity["is_bot"], user_activity["is_anomaly"])
cm_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Anomaly"],
    columns=["Predicted Normal", "Predicted Anomaly"]
)
st.markdown("#### Confusion Matrix")
st.table(cm_df)
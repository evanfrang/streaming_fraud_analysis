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

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.2rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

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

@st.cache_data
def load_time_series():
    return pd.read_pickle("data/time_series.pkl")

user_activity = load_user_activity()
user_days = load_user_days()
time_series = load_time_series()

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs([
    "Overview", "Anomaly Detection",
    "Time Series"
])

# -------------------------
# Tab 1: Overview
# -------------------------
with tab1:

    numeric_cols = [
        col for col in user_activity.select_dtypes(include="number").columns
        if col != "user_id"
    ]

    selected_metric = st.selectbox("Filter metric", numeric_cols, index=0, key="metric_overview")
    threshold = st.number_input(f"Minimum {selected_metric}", value=0.0, key="thresh_overview")

    sus_users = user_activity[user_activity[selected_metric] >= threshold]

    st.write(f"### {len(sus_users)} Suspicious Users (filtered by {selected_metric} ≥ {round(threshold,3)})")

    styled = sus_users.sort_values(selected_metric, ascending=False).set_index("user_id").style.format(
        precision=2
    ).background_gradient(cmap="Blues", subset=[selected_metric])

    st.dataframe(styled, use_container_width=True, height=500)

    c1, c2 = st.columns(2)

    with c1:
        fig_all = px.histogram(user_activity, x=selected_metric, nbins=50,
                               title=f"Distribution of {selected_metric} (All Users)")
        st.plotly_chart(fig_all, use_container_width=True)

    with c2:
        fig_sus = px.histogram(sus_users, x=selected_metric, nbins=50,
                               title=f"Distribution of {selected_metric} (Suspicious Only)")
        st.plotly_chart(fig_sus, use_container_width=True)

    st.write("## Explore a user")
    user_choice = st.selectbox("Pick a user_id", sus_users["user_id"].unique())

    if user_choice:
        try:
            # Daily Streams
            user_days_sel = user_days[user_days["user_id"] == user_choice]
            fig2 = px.line(user_days_sel, x="timestamp", y="daily_streams", 
                           title=f"Daily streams for User {user_choice}")
            fig3 = px.line(user_days_sel, x="timestamp", y="listen_pct", 
                           title=f"Daily Listening % for User {user_choice}")

            # Autocorrelation
            user_series = (
                user_days_sel.set_index('timestamp')
                .asfreq('D', fill_value=0)['daily_streams']
            )
            ac = np.correlate(user_series - user_series.mean(),
                              user_series - user_series.mean(), mode='full')
            ac = ac[ac.size // 2:]  
            ac /= ac[0]
            lags = np.arange(1, len(ac))  
            ac = ac[1:]
            fig4 = px.line(x=lags, y=ac,
                           labels={'x': 'Lag (days)', 'y': 'Autocorrelation'},
                           title=f"Autocorrelation for User {user_choice}")

            # Fourier Transform
            values = user_series.values - user_series.mean()
            n = len(values)
            fft_vals = np.fft.fft(values)
            freqs = np.fft.fftfreq(n, d=1)
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            power = np.abs(fft_vals[pos_mask])**2
            fig5 = px.line(x=freqs, y=power,
                           labels={'x': 'Frequency (cycles per day)', 'y': 'Power'},
                           title=f"Fourier Transform Spectrum for User {user_choice}")

            c1, c2 = st.columns(2)
            c1.plotly_chart(fig2, use_container_width=True)
            c2.plotly_chart(fig3, use_container_width=True)

            c3, c4 = st.columns(2)
            c3.plotly_chart(fig4, use_container_width=True)
            c4.plotly_chart(fig5, use_container_width=True)

        except FileNotFoundError:
            st.info("Per-day data not available — only summary features shown.")
            st.write(user_activity[user_activity["user_id"] == user_choice].T)


# -------------------------
# Tab 4: ML Anomaly Detection
# -------------------------
with tab2:
    st.write("### Users flagged by Isolation Forest")

    features = user_activity.select_dtypes(include="number").columns.tolist()
    X = user_activity[features].drop(columns=["user_id"])

    iso = IsolationForest(n_estimators=300, contamination=0.08, random_state=13)
    iso.fit(X)

    user_activity["anomaly_score"] = iso.decision_function(X)
    user_activity["is_anomaly"] = iso.predict(X)
    user_activity["is_bot"] = user_activity['user_id'] >= 301  # dataset quirk

    ml_anomalies = user_activity[user_activity["is_anomaly"] == -1].copy()

    st.dataframe(
        ml_anomalies.sort_values("anomaly_score").set_index("user_id"),
        use_container_width=True,
        height=500
    )

    user_activity["is_anomaly"] = user_activity["is_anomaly"].apply(lambda x: 1 if x==-1 else 0)
    user_activity["is_bot"] = user_activity["is_bot"].astype(int)

    st.markdown("#### Isolation Forest Performance")

    prec, rec, f1, _ = precision_recall_fscore_support(
        user_activity["is_bot"], user_activity["is_anomaly"], average=None
    )

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

# -------------------------
# Tab 5: Time Series
# -------------------------

with tab3:
    st.subheader("Time Series Rule-Based Detection")

    threshold_val = st.number_input("Stream threshold", min_value=1, value=100, step=10)
    consec_days = st.number_input("Consecutive days above threshold", min_value=1, value=3, step=1)

    time_series["day_date"] = pd.to_datetime(time_series["day_date"])

    fig = px.scatter(time_series, x="day_date", y="y", title="Time Series with Mean & Bounds")
    fig.update_traces(marker=dict(symbol="x", size=7, color="black"))

    # Add the bounds as a shaded band
    fig.add_scatter(
        x=pd.concat([time_series["day_date"], time_series["day_date"][::-1]]),
        y=pd.concat([time_series["lower_1"], time_series["upper_1"][::-1]]),
        fill="toself",
        fillcolor="rgba(0,80,200,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Bound 1x"
    )

    fig.add_scatter(
        x=pd.concat([time_series["day_date"], time_series["day_date"][::-1]]),
        y=pd.concat([time_series["lower_2"], time_series["upper_2"][::-1]]),
        fill="toself",
        fillcolor="rgba(0,200,80,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Bound 2x"
    )

    fig.add_scatter(
        x=time_series["day_date"],
        y=time_series["y_base"],
        mode="lines",
        line=dict(color="red"),
        name="GP Mean"
    )

    fig.update_layout(
        height=600,  # adjust height in pixels
        margin=dict(l=40, r=40, t=40, b=40),
        title=dict(text="Artist 1 Streams", font=dict(size=22)),
        xaxis=dict(title=dict(text="Days",font=dict(size=18)),
            tickfont=dict(size=14)
        ),
        yaxis=dict(title=dict(text="Streams",font=dict(size=18)),
            tickfont=dict(size=14)
        ),
        legend=dict(font=dict(size=16))
    )

    st.plotly_chart(fig, use_container_width=True)
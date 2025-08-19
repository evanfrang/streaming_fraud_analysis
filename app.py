import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

@st.cache_data
def load_user_activity():
    return pd.read_pickle("user_activity.pkl")

@st.cache_data
def load_user_days():
    return pd.read_pickle("user_day_counts.pkl")

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
    index=numeric_cols.index("total_streams") if "total_streams" in numeric_cols else 0
)

threshold = st.sidebar.number_input(f"Minimum {selected_metric}", value=60.0)

sus_users = user_activity[user_activity[selected_metric] >= threshold]

st.write(f"### {len(sus_users)} Suspicious Users (filtered by {selected_metric} ≥ {round(threshold,3)})")

# -------------------------
# Leaderboard
# -------------------------
st.dataframe(
    sus_users.sort_values(selected_metric, ascending=False)
)

# -------------------------
# Distribution plot
# -------------------------
fig = px.histogram(user_activity, x=selected_metric, nbins=50, title=f"Distribution of {selected_metric}")
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Per-user drill down
# -------------------------
st.write("## Explore a user")
user_choice = st.selectbox("Pick a user_id", sus_users["user_id"].unique())

if user_choice:
    try:
        # Daily Streams
        user_day_counts = pd.read_pickle("user_day_counts.pkl")
        user_day_counts = user_day_counts[user_day_counts["user_id"] == user_choice]
        fig2 = px.line(user_day_counts, x="timestamp", y="daily_streams", title=f"Daily streams for {user_choice}")
        st.plotly_chart(fig2, use_container_width=True)

        # Listening %
        fig3 = px.line(user_day_counts, x="timestamp", y="listen_pct", title=f"Daily Listening % for {user_choice}")
        st.plotly_chart(fig3, use_container_width=True)

        # Autocorrelation
        user_series = (
            user_day_counts[user_day_counts['user_id'] == user_choice]
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
            title=f"Autocorrelation for {user_choice}"
        )
        st.plotly_chart(fig4, use_container_width=True)

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
            title=f"Fourier Transform Spectrum for {user_choice}"
        )
        st.plotly_chart(fig5, use_container_width=True)

    except FileNotFoundError:
        st.info("Per-day data not available — only summary features shown.")
        st.write(user_activity[user_activity["user_id"] == user_choice].T)
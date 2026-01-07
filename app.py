import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import pytz

# -----------------------------------------------------------------------------
# APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gold Hourly Stats (NY Time)",
    page_icon="🏆",
    layout="wide"
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Cache data for 1 hour to prevent API spam
def get_gold_data(period_days):
    """
    Fetches hourly data for Gold Futures (GC=F) from Yahoo Finance.
    Limit for hourly data in yfinance is typically around 730 days.
    """
    # GC=F is Gold Futures. You could also use 'XAUUSD=X' for Spot Gold.
    ticker = "GC=F"
    
    # Calculate period string (e.g., "730d")
    period_str = f"{period_days}d"
    
    # Fetch data
    df = yf.download(ticker, interval="1h", period=period_str, progress=False)
    
    # Handle MultiIndex columns if present (common in newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to make Datetime a column for easier processing
    df = df.reset_index()
    
    # Ensure standard column names
    df.columns = [c.capitalize() for c in df.columns]
    
    return df

def process_data(df):
    """
    Converts timestamps to NY time and identifies the hour of HOD/LOD.
    """
    # Rename 'Datetime' or 'Date' to 'timestamp'
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'timestamp'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'timestamp'})
    else:
        st.error("Could not find datetime column in data.")
        return None

    # 1. Convert Timezone to America/New_York
    # Yahoo Finance usually returns UTC. We assume UTC if tz-naive.
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')

    # 2. Create Date and Hour columns based on NY time
    df['Date_NY'] = df['timestamp'].dt.date
    df['Hour_NY'] = df['timestamp'].dt.hour

    # 3. Identify High of Day (HOD) and Low of Day (LOD) Hour
    # Group by NY Date
    grouped = df.groupby('Date_NY')

    stats_list = []

    for date, group in grouped:
        # Skip incomplete days (e.g. strict checking can be added here)
        if len(group) < 5: 
            continue

        # Find the row with the max High
        id_max = group['High'].idxmax()
        hour_high = group.loc[id_max, 'Hour_NY']

        # Find the row with the min Low
        id_min = group['Low'].idxmin()
        hour_low = group.loc[id_min, 'Hour_NY']

        stats_list.append({
            'Date': date,
            'HOD_Hour': hour_high,
            'LOD_Hour': hour_low
        })

    return pd.DataFrame(stats_list)

# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------
st.title("🏆 Gold Hourly Volatility Analyzer")
st.markdown("""
**Objective:** Detect statistically which hour of the day produces the **High** and **Low** of the day.
*Timezone:* **New York (EST/EDT)** | *Asset:* **Gold Futures (GC=F)**
""")

# Sidebar Controls
st.sidebar.header("Settings")
days_history = st.sidebar.slider("Days of History", min_value=30, max_value=720, value=365, step=30)
st.sidebar.info("Note: Hourly data is limited to the last ~730 days by the data provider.")

# Load Data
with st.spinner("Fetching market data..."):
    raw_df = get_gold_data(days_history)

if raw_df is not None and not raw_df.empty:
    # Process Data
    stats_df = process_data(raw_df)
    
    if stats_df is not None and not stats_df.empty:
        
        # ---------------------------------------------------------------------
        # CALCULATION: FREQUENCY
        # ---------------------------------------------------------------------
        total_days = len(stats_df)
        
        # Count frequency of HOD per hour
        hod_counts = stats_df['HOD_Hour'].value_counts().sort_index()
        hod_pct = (hod_counts / total_days) * 100
        
        # Count frequency of LOD per hour
        lod_counts = stats_df['LOD_Hour'].value_counts().sort_index()
        lod_pct = (lod_counts / total_days) * 100

        # Combine into a single DataFrame for charting
        chart_df = pd.DataFrame({
            'Hour (NY Time)': range(0, 24),
        }).set_index('Hour (NY Time)')

        chart_df['High Probability (%)'] = hod_pct
        chart_df['Low Probability (%)'] = lod_pct
        chart_df = chart_df.fillna(0) # Fill hours with 0 occurrences

        # ---------------------------------------------------------------------
        # VISUALIZATION
        # ---------------------------------------------------------------------
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Days Analyzed", total_days)
        
        top_hod_hour = chart_df['High Probability (%)'].idxmax()
        col2.metric("Most Frequent HOD Hour", f"{top_hod_hour}:00 NY", f"{chart_df.loc[top_hod_hour, 'High Probability (%)']:.1f}%")
        
        top_lod_hour = chart_df['Low Probability (%)'].idxmax()
        col3.metric("Most Frequent LOD Hour", f"{top_lod_hour}:00 NY", f"{chart_df.loc[top_lod_hour, 'Low Probability (%)']:.1f}%")

        st.markdown("---")

        # Charts
        tab1, tab2 = st.tabs(["📊 High of Day Analysis", "📉 Low of Day Analysis"])

        with tab1:
            st.subheader("Probability of Forming the High of the Day (by Hour)")
            fig_high = px.bar(
                chart_df, 
                x=chart_df.index, 
                y='High Probability (%)',
                labels={'x': 'Hour of Day (NY Time)', 'High Probability (%)': 'Probability (%)'},
                color='High Probability (%)',
                color_continuous_scale='Reds'
            )
            fig_high.update_layout(xaxis=dict(tickmode='linear', dtick=1))
            st.plotly_chart(fig_high, use_container_width=True)

        with tab2:
            st.subheader("Probability of Forming the Low of the Day (by Hour)")
            fig_low = px.bar(
                chart_df, 
                x=chart_df.index, 
                y='Low Probability (%)',
                labels={'x': 'Hour of Day (NY Time)', 'Low Probability (%)': 'Probability (%)'},
                color='Low Probability (%)',
                color_continuous_scale='Greens'
            )
            fig_low.update_layout(xaxis=dict(tickmode='linear', dtick=1))
            st.plotly_chart(fig_low, use_container_width=True)

        # ---------------------------------------------------------------------
        # HEATMAP VIEW (Optional advanced view)
        # ---------------------------------------------------------------------
        with st.expander("View Raw Frequency Data"):
            st.dataframe(chart_df.style.format("{:.1f}%").background_gradient(cmap="Blues"))

    else:
        st.warning("Not enough data to calculate statistics.")
else:
    st.error("Failed to download data. Please try again later or reduce the history range.")

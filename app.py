import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import pytz
from datetime import datetime

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
@st.cache_data(ttl=3600)
def get_gold_data(period_days):
    """
    Fetches hourly data. Tries Spot Gold (XAUUSD=X) first, then Futures (GC=F).
    """
    period_str = f"{period_days}d"
    tickers = ["XAUUSD=X", "GC=F"]
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, interval="1h", period=period_str, progress=False)
            
            if df.empty:
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            df.columns = [c.capitalize() for c in df.columns]
            
            if 'High' in df.columns and 'Low' in df.columns:
                return df, ticker

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue
            
    return pd.DataFrame(), None

def process_data(df):
    """
    Converts timestamps to NY time and identifies the hour of HOD/LOD.
    """
    if df is None or df.empty:
        return None

    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'timestamp'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'timestamp'})
    else:
        return None

    # 1. Convert Timezone to America/New_York
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')

    # 2. Create Date and Hour columns based on NY time
    df['Date_NY'] = df['timestamp'].dt.date
    df['Hour_NY'] = df['timestamp'].dt.hour

    # 3. Identify High of Day (HOD) and Low of Day (LOD) Hour
    grouped = df.groupby('Date_NY')

    stats_list = []

    for date, group in grouped:
        if len(group) < 5: 
            continue

        id_max = group['High'].idxmax()
        hour_high = group.loc[id_max, 'Hour_NY']

        id_min = group['Low'].idxmin()
        hour_low = group.loc[id_min, 'Hour_NY']

        # Get Day Name (Monday, Tuesday, etc.)
        day_name = date.strftime('%A')

        stats_list.append({
            'Date': date,
            'Day_Name': day_name,
            'HOD_Hour': hour_high,
            'LOD_Hour': hour_low
        })

    return pd.DataFrame(stats_list)

def calculate_frequencies(df):
    """
    Calculates the probability % for each hour.
    """
    total = len(df)
    if total == 0:
        return pd.DataFrame()

    hod_counts = df['HOD_Hour'].value_counts().sort_index()
    hod_pct = (hod_counts / total) * 100
    
    lod_counts = df['LOD_Hour'].value_counts().sort_index()
    lod_pct = (lod_counts / total) * 100

    chart_df = pd.DataFrame({'Hour (NY Time)': range(0, 24)}).set_index('Hour (NY Time)')
    chart_df['High Probability (%)'] = hod_pct
    chart_df['Low Probability (%)'] = lod_pct
    chart_df = chart_df.fillna(0)
    
    return chart_df

# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------
st.title("🏆 Gold Hourly Volatility Analyzer")
st.markdown("""
**Objective:** Detect statistically which hour of the day produces the **High** and **Low** of the day.
*Timezone:* **New York (EST/EDT)**
""")

# Sidebar
st.sidebar.header("Settings")
days_history = st.sidebar.slider("Days of History", min_value=30, max_value=720, value=60, step=30)
st.sidebar.caption("Data provided by Yahoo Finance.")

# Load Data
with st.spinner("Fetching market data..."):
    raw_df, active_ticker = get_gold_data(days_history)

if raw_df is not None and not raw_df.empty:
    st.sidebar.success(f"Loaded data for: {active_ticker}")
    stats_df = process_data(raw_df)
    
    if stats_df is not None and not stats_df.empty:
        
        # Determine Current NY Day
        ny_tz = pytz.timezone('America/New_York')
        today_ny = datetime.now(ny_tz).strftime('%A')

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Overall Stats", "📅 Daily Forecast (Day Specific)", "💾 Raw Data"])

        # --- TAB 1: OVERALL ---
        with tab1:
            st.subheader("All Days Combined")
            chart_df_all = calculate_frequencies(stats_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**High of Day Probability**")
                fig_high = px.bar(chart_df_all, x=chart_df_all.index, y='High Probability (%)', 
                                  color='High Probability (%)', color_continuous_scale='Reds')
                st.plotly_chart(fig_high, use_container_width=True)

            with col2:
                st.write("**Low of Day Probability**")
                fig_low = px.bar(chart_df_all, x=chart_df_all.index, y='Low Probability (%)', 
                                 color='Low Probability (%)', color_continuous_scale='Greens')
                st.plotly_chart(fig_low, use_container_width=True)

        # --- TAB 2: DAY SPECIFIC ---
        with tab2:
            st.subheader("Day-Specific Analysis")
            
            # Day Selector
            selected_day = st.selectbox("Select Day of Week:", 
                                        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], 
                                        index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'].index(today_ny) if today_ny in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] else 0)
            
            # Filter Data
            day_stats_df = stats_df[stats_df['Day_Name'] == selected_day]
            day_count = len(day_stats_df)
            
            if day_count > 0:
                st.info(f"Analyzing **{day_count}** recorded {selected_day}s in the dataset.")
                
                chart_df_day = calculate_frequencies(day_stats_df)
                
                # Metrics for that day
                top_hod_hour = chart_df_day['High Probability (%)'].idxmax()
                top_lod_hour = chart_df_day['Low Probability (%)'].idxmax()
                
                m1, m2 = st.columns(2)
                m1.metric(f"Best Time for High ({selected_day})", f"{top_hod_hour}:00 NY", f"{chart_df_day.loc[top_hod_hour, 'High Probability (%)']:.1f}% prob")
                m2.metric(f"Best Time for Low ({selected_day})", f"{top_lod_hour}:00 NY", f"{chart_df_day.loc[top_lod_hour, 'Low Probability (%)']:.1f}% prob")

                # Charts
                c1, c2 = st.columns(2)
                with c1:
                    fig_d_high = px.bar(chart_df_day, x=chart_df_day.index, y='High Probability (%)',
                                        title=f"High of Day Probability on {selected_day}s",
                                        color='High Probability (%)', color_continuous_scale='Reds')
                    st.plotly_chart(fig_d_high, use_container_width=True)
                
                with c2:
                    fig_d_low = px.bar(chart_df_day, x=chart_df_day.index, y='Low Probability (%)',
                                       title=f"Low of Day Probability on {selected_day}s",
                                       color='Low Probability (%)', color_continuous_scale='Greens')
                    st.plotly_chart(fig_d_low, use_container_width=True)
            else:
                st.warning(f"No data found for {selected_day}. Try increasing history duration.")

        # --- TAB 3: RAW DATA ---
        with tab3:
             st.dataframe(stats_df.style.format("{:.1f}"), use_container_width=True)

    else:
        st.warning("Data downloaded but processing failed.")
else:
    st.error("❌ Failed to download data. Try reducing the history range.")

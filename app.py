import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import time
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------
# APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pro Trader Analytics Suite",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def format_hour_ampm(hour_int):
    if hour_int == 0: return "12:00 AM"
    elif hour_int == 12: return "12:00 PM"
    elif hour_int < 12: return f"{hour_int}:00 AM"
    else: return f"{hour_int - 12}:00 PM"

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data(ticker, period_days):
    """
    Robust data fetching with Headers, Retries, and Fallbacks.
    """
    period_str = f"{period_days}d"
    
    # Custom Session with Headers to avoid 403/404 errors
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })

    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use the custom session
            ticker_obj = yf.Ticker(ticker, session=session)
            df = ticker_obj.history(period=period_str, interval="1h")
            
            if df.empty:
                # If empty, wait and try again
                time.sleep(1)
                continue

            # CLEAN DATA
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            
            # Standardize Date Column
            col_map = {c: c.capitalize() for c in df.columns}
            df = df.rename(columns=col_map)
            
            if 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'timestamp'})
            elif 'Date' in df.columns:
                df = df.rename(columns={'Date': 'timestamp'})
            
            # Validate Columns
            required = ['timestamp', 'High', 'Low', 'Open', 'Close']
            # Sometimes 'timestamp' is 'Timestamp' (capitalized)
            if 'Timestamp' in df.columns: 
                df = df.rename(columns={'Timestamp': 'timestamp'})

            if all(col in df.columns for col in required):
                return df
                
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)
            
    return pd.DataFrame()

def process_data(df):
    if df is None or df.empty: return None, None

    # Timezone Conversion
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')

    df['Date_NY'] = df['timestamp'].dt.date
    df['Hour_NY'] = df['timestamp'].dt.hour
    df['Day_Name'] = df['timestamp'].dt.strftime('%A')
    df['Range'] = df['High'] - df['Low']

    grouped = df.groupby('Date_NY')
    stats_list = []

    for date, group in grouped:
        if len(group) < 4: continue 

        id_max = group['High'].idxmax()
        hour_high = group.loc[id_max, 'Hour_NY']
        price_high = group.loc[id_max, 'High']

        id_min = group['Low'].idxmin()
        hour_low = group.loc[id_min, 'Hour_NY']
        price_low = group.loc[id_min, 'Low']

        daily_range = price_high - price_low

        stats_list.append({
            'Date': date,
            'Day_Name': group.iloc[0]['Day_Name'],
            'HOD_Hour': hour_high,
            'LOD_Hour': hour_low,
            'Daily_Range': daily_range,
            'Open_Price': group.iloc[0]['Open'],
            'Close_Price': group.iloc[-1]['Close']
        })

    return pd.DataFrame(stats_list), df

def train_hod_lod_model(stats_df):
    if stats_df is None or len(stats_df) < 10: return None, 0
    ml_df = stats_df.copy()
    ml_df['Target_AM_High'] = ml_df['HOD_Hour'].apply(lambda x: 1 if 9 <= x <= 11 else 0)
    ml_df['Day_Code'] = pd.Categorical(ml_df['Day_Name']).codes
    ml_df['Prev_Range'] = ml_df['Daily_Range'].shift(1)
    ml_df = ml_df.dropna()
    if len(ml_df) < 20: return None, 0

    X = ml_df[['Day_Code', 'Prev_Range']]
    y = ml_df['Target_AM_High']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Control Center")

# Data Source Selector
data_source = st.sidebar.radio("Data Source", ["Live API (Yahoo)", "Upload CSV (Backup)"])

stats_df = None
hourly_df = None
selected_asset_name = "Custom Data"

if data_source == "Live API (Yahoo)":
    asset_map = {
        "Gold (Spot)": "XAUUSD=X",
        "Gold (Futures)": "GC=F",
        "Gold ETF (Backup)": "GLD",  # Added GLD as backup
        "Nasdaq 100": "NQ=F",
        "S&P 500": "ES=F",
        "Euro / USD": "EURUSD=X"
    }
    selected_asset_name = st.sidebar.selectbox("Select Asset", list(asset_map.keys()))
    selected_ticker = asset_map[selected_asset_name]
    days_history = st.sidebar.slider("Days of History", 30, 720, 180, 30)

    with st.spinner(f"Connecting to Exchange ({selected_ticker})..."):
        raw_df = get_market_data(selected_ticker, days_history)
        if not raw_df.empty:
            stats_df, hourly_df = process_data(raw_df)
        else:
            st.error("⚠️ Connection Blocked or No Data.")
            st.info("Tip: Try selecting 'Gold ETF (Backup)' if Futures/Spot are failing.")

elif data_source == "Upload CSV (Backup)":
    st.sidebar.info("Upload a CSV from TradingView with columns: time, open, high, low, close")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            # Basic Cleaning for standard TradingView exports
            raw_df.columns = [c.lower() for c in raw_df.columns]
            if 'time' in raw_df.columns: raw_df = raw_df.rename(columns={'time': 'timestamp'})
            # Ensure datetime
            raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
            # Capitalize for processor
            raw_df = raw_df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close'})
            
            stats_df, hourly_df = process_data(raw_df)
            st.sidebar.success("CSV Processed Successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

# -----------------------------------------------------------------------------
# DASHBOARD RENDERING
# -----------------------------------------------------------------------------
if stats_df is not None and not stats_df.empty:
    st.title(f"🦅 {selected_asset_name} Statistical Profile")
    st.caption(f"Timezone: New York (EST) | Data Points: {len(stats_df)} Days")

    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)
    today_name = now_ny.strftime('%A')
    
    avg_range = stats_df['Daily_Range'].mean()
    most_volatile_day = stats_df.groupby('Day_Name')['Daily_Range'].mean().idxmax()
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Current Day", today_name, now_ny.strftime("%I:%M %p"))
    kpi2.metric("Avg Daily Range", f"{avg_range:.2f}")
    kpi3.metric("Most Volatile Day", most_volatile_day)
    kpi4.metric("Days Analyzed", len(stats_df))

    tab_forecast, tab_analytics, tab_ml, tab_tools = st.tabs([
        "📅 Daily Forecast", "📊 Deep Analytics", "🤖 AI Insights", "🛠️ Tools & Export"
    ])

    # === TAB 1: DAILY FORECAST ===
    with tab_forecast:
        st.subheader(f"Session Profile: {today_name}")
        day_stats = stats_df[stats_df['Day_Name'] == today_name]
        
        if not day_stats.empty:
            hod_counts = day_stats['HOD_Hour'].value_counts(normalize=True).sort_index() * 100
            lod_counts = day_stats['LOD_Hour'].value_counts(normalize=True).sort_index() * 100
            
            chart_data = pd.DataFrame({'Hour': range(24)}).set_index('Hour')
            chart_data['High Prob %'] = hod_counts
            chart_data['Low Prob %'] = lod_counts
            chart_data = chart_data.fillna(0)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['High Prob %'], name='High Probability', marker_color='#ff4b4b'))
            fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['Low Prob %'], name='Low Probability', marker_color='#2bd27f'))

            kill_zones = [(2, 5, "London Open", "rgba(0,0,255,0.1)"), (7, 10, "NY Open", "rgba(255,165,0,0.1)"), (13, 16, "NY PM", "rgba(128,0,128,0.1)")]
            for start, end, label, color in kill_zones:
                fig.add_vrect(x0=start-0.5, x1=end-0.5, fillcolor=color, opacity=1, layer="below", line_width=0, annotation_text=label)

            fig.update_layout(title=f"Hourly Probabilities for {today_name}", xaxis_title="Hour (NY Time)", yaxis_title="Probability (%)", barmode='group', hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            top_h = chart_data['High Prob %'].idxmax()
            top_l = chart_data['Low Prob %'].idxmax()
            st.info(f"💡 **Strategy Note:** On {today_name}s, look for **Shorts around {format_hour_ampm(top_h)}** and **Longs around {format_hour_ampm(top_l)}**.")
        else:
            st.warning("Not enough data for this specific day.")

    # === TAB 2: DEEP ANALYTICS ===
    with tab_analytics:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Volatility Heatmap (Avg Range)")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            vol_stats = stats_df.groupby('Day_Name')['Daily_Range'].mean().reindex(day_order)
            fig_vol = px.bar(x=vol_stats.index, y=vol_stats.values, color=vol_stats.values, color_continuous_scale='Viridis', title="Which Day Moves the Most?")
            st.plotly_chart(fig_vol, use_container_width=True)
        with col2:
            st.subheader("Trend vs Reversal Days")
            bullish_days = len(stats_df[stats_df['Close_Price'] > stats_df['Open_Price']])
            bearish_days = len(stats_df[stats_df['Close_Price'] < stats_df['Open_Price']])
            fig_pie = px.pie(values=[bullish_days, bearish_days], names=['Bullish Days', 'Bearish Days'], color_discrete_sequence=['#2bd27f', '#ff4b4b'], hole=0.4, title="Daily Bias Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

    # === TAB 3: AI INSIGHTS ===
    with tab_ml:
        st.subheader("🤖 Pattern Recognition Engine")
        if st.button("Train AI Model"):
            model, accuracy = train_hod_lod_model(stats_df)
            if model:
                st.success(f"Model Accuracy: **{accuracy:.1%}**")
                last_range = stats_df.iloc[-1]['Daily_Range']
                day_code = (datetime.now().weekday() + 1) % 7
                prediction = model.predict([[day_code, last_range]])[0]
                prob = model.predict_proba([[day_code, last_range]])[0]
                st.markdown("### 🔮 Tomorrow's Prediction")
                if prediction == 1: st.success(f"High chance ({prob[1]:.1f}%) of High forming in AM Session (9-11 AM).")
                else: st.warning(f"Low chance ({prob[0]:.1f}%) of AM High. Expect High later in the day.")

    # === TAB 4: TOOLS ===
    with tab_tools:
        st.subheader("💻 Pine Script Generator")
        best_h_hour = stats_df['HOD_Hour'].mode()[0]
        best_l_hour = stats_df['LOD_Hour'].mode()[0]
        st.code(f"//@version=5\nindicator('Stats High/Low', overlay=true)\nhighTime = {best_h_hour}\nlowTime = {best_l_hour}\nbgcolor(hour==highTime ? color.new(color.red,80) : na)\nbgcolor(hour==lowTime ? color.new(color.green,80) : na)", language="pine")
        st.download_button("Download CSV", data=stats_df.to_csv(index=False).encode('utf-8'), file_name=f'{selected_asset_name}_stats.csv', mime='text/csv')

else:
    st.warning("waiting for data...")

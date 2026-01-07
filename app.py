import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
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

@st.cache_data(ttl=3600)
def get_market_data(ticker, period_days):
    """
    Robust data fetching with strict column cleaning.
    """
    period_str = f"{period_days}d"
    try:
        # Fetch data
        df = yf.download(ticker, interval="1h", period=period_str, progress=False)
        
        if df.empty:
            return pd.DataFrame()

        # -------------------------------------------------------
        # CRITICAL FIX: Handle MultiIndex Columns (Price, Ticker)
        # -------------------------------------------------------
        # yfinance often returns columns like: ('Close', 'GC=F')
        if isinstance(df.columns, pd.MultiIndex):
            # We want the level that contains 'Close', 'High', etc.
            # Usually that is level 0.
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to make 'Datetime' a column
        df = df.reset_index()

        # Rename the date column to 'timestamp' standard
        # yfinance might call it 'Date' or 'Datetime'
        col_map = {c: c.capitalize() for c in df.columns} # Capitalize everything first
        df = df.rename(columns=col_map)
        
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'timestamp'})
        elif 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})
            
        # Ensure we have the required columns
        required = ['Timestamp', 'High', 'Low', 'Open', 'Close']
        # Note: We capitalized columns above, so we check for capitalized names
        # 'timestamp' is lowercase in our logic, so let's fix that:
        df = df.rename(columns={'Timestamp': 'timestamp'}) 
        
        if 'timestamp' not in df.columns:
            # Last ditch effort to find the date column
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df = df.rename(columns={col: 'timestamp'})
                    break

        return df
    except Exception as e:
        st.error(f"Data Download Error: {e}")
        return pd.DataFrame()

def process_data(df):
    if df is None or df.empty: return None, None

    # Check for required columns again
    req_cols = ['timestamp', 'High', 'Low']
    for c in req_cols:
        if c not in df.columns:
            st.error(f"Missing column: {c}. Available: {df.columns.tolist()}")
            return None, None

    # Timezone Conversion (UTC -> NY)
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    # Convert to NY time
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')

    # Derived Columns
    df['Date_NY'] = df['timestamp'].dt.date
    df['Hour_NY'] = df['timestamp'].dt.hour
    df['Day_Name'] = df['timestamp'].dt.strftime('%A')
    
    # Calculate Range
    df['Range'] = df['High'] - df['Low']

    # --- AGGREGATION PER DAY ---
    grouped = df.groupby('Date_NY')
    stats_list = []

    for date, group in grouped:
        if len(group) < 5: continue # Skip incomplete days

        # High/Low stats
        id_max = group['High'].idxmax()
        hour_high = group.loc[id_max, 'Hour_NY']
        price_high = group.loc[id_max, 'High']

        id_min = group['Low'].idxmin()
        hour_low = group.loc[id_min, 'Hour_NY']
        price_low = group.loc[id_min, 'Low']

        # Daily Range (Volatility)
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

# -----------------------------------------------------------------------------
# MACHINE LEARNING ENGINE
# -----------------------------------------------------------------------------
def train_hod_lod_model(stats_df):
    if stats_df is None or len(stats_df) < 10:
        return None, 0

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
# Sidebar Configuration
st.sidebar.title("🎛️ Control Center")

asset_map = {
    "Gold (Spot)": "XAUUSD=X",
    "Gold (Futures)": "GC=F",
    "Nasdaq 100": "NQ=F",
    "S&P 500": "ES=F",
    "Euro / USD": "EURUSD=X"
}

selected_asset_name = st.sidebar.selectbox("Select Asset", list(asset_map.keys()))
selected_ticker = asset_map[selected_asset_name]

days_history = st.sidebar.slider("Days of History", 30, 720, 180, 30)

# Fetch Data
stats_df = None
hourly_df = None

with st.spinner(f"Analyzing {selected_asset_name} market structure..."):
    raw_df = get_market_data(selected_ticker, days_history)
    
    if not raw_df.empty:
        stats_df, hourly_df = process_data(raw_df)
    else:
        # Fallback debug message
        st.error("⚠️ Data download returned empty.")
        with st.expander("Debug Info"):
            st.write(f"Ticker: {selected_ticker}")
            st.write("Troubleshooting: Try reducing 'Days of History' or switching Assets.")

if stats_df is not None and not stats_df.empty:
    # -------------------------------------------------------------------------
    # DASHBOARD HEADER
    # -------------------------------------------------------------------------
    st.title(f"🦅 {selected_asset_name} Statistical Profile")
    st.caption(f"Timezone: New York (EST) | Data Points: {len(stats_df)} Days")

    # Current Day Context
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)
    today_name = now_ny.strftime('%A')
    
    # KPIs
    avg_range = stats_df['Daily_Range'].mean()
    most_volatile_day = stats_df.groupby('Day_Name')['Daily_Range'].mean().idxmax()
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Current Day", today_name, now_ny.strftime("%I:%M %p"))
    kpi2.metric("Avg Daily Range", f"{avg_range:.2f}")
    kpi3.metric("Most Volatile Day", most_volatile_day)
    kpi4.metric("Days Analyzed", len(stats_df))

    # -------------------------------------------------------------------------
    # MAIN TABS
    # -------------------------------------------------------------------------
    tab_forecast, tab_analytics, tab_ml, tab_tools = st.tabs([
        "📅 Daily Forecast", 
        "📊 Deep Analytics", 
        "🤖 AI Insights", 
        "🛠️ Tools & Export"
    ])

    # === TAB 1: DAILY FORECAST ===
    with tab_forecast:
        st.subheader(f"Session Profile: {today_name}")
        
        # Filter for current day
        day_stats = stats_df[stats_df['Day_Name'] == today_name]
        
        if not day_stats.empty:
            # Calculate Probabilities
            hod_counts = day_stats['HOD_Hour'].value_counts(normalize=True).sort_index() * 100
            lod_counts = day_stats['LOD_Hour'].value_counts(normalize=True).sort_index() * 100
            
            chart_data = pd.DataFrame({'Hour': range(24)}).set_index('Hour')
            chart_data['High Prob %'] = hod_counts
            chart_data['Low Prob %'] = lod_counts
            chart_data = chart_data.fillna(0)

            # --- ADVANCED CHART WITH KILL ZONES ---
            fig = go.Figure()

            # Add Bars
            fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['High Prob %'], name='High Probability', marker_color='#ff4b4b'))
            fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['Low Prob %'], name='Low Probability', marker_color='#2bd27f'))

            # Add ICT Kill Zones
            kill_zones = [
                (2, 5, "London Open", "rgba(0, 0, 255, 0.1)"),
                (7, 10, "NY Open", "rgba(255, 165, 0, 0.1)"),
                (13, 16, "NY PM", "rgba(128, 0, 128, 0.1)")
            ]
            
            for start, end, label, color in kill_zones:
                fig.add_vrect(
                    x0=start-0.5, x1=end-0.5, 
                    fillcolor=color, opacity=1, layer="below", line_width=0,
                    annotation_text=label, annotation_position="top left"
                )

            fig.update_layout(
                title=f"Hourly Probabilities for {today_name} (with Kill Zones)",
                xaxis_title="Hour (NY Time)",
                yaxis_title="Probability (%)",
                barmode='group',
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Best Times Summary
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
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            vol_stats = stats_df.groupby('Day_Name')['Daily_Range'].mean().reindex(day_order)
            
            fig_vol = px.bar(
                x=vol_stats.index, y=vol_stats.values,
                color=vol_stats.values, color_continuous_scale='Viridis',
                labels={'x': 'Day', 'y': 'Avg Range (Price)'},
                title="Which Day Moves the Most?"
            )
            st.plotly_chart(fig_vol, use_container_width=True)
            
        with col2:
            st.subheader("Trend vs Reversal Days")
            # Logic: If Close > Open = Bullish.
            bullish_days = len(stats_df[stats_df['Close_Price'] > stats_df['Open_Price']])
            bearish_days = len(stats_df[stats_df['Close_Price'] < stats_df['Open_Price']])
            
            fig_pie = px.pie(
                values=[bullish_days, bearish_days], 
                names=['Bullish Days (Green)', 'Bearish Days (Red)'],
                color_discrete_sequence=['#2bd27f', '#ff4b4b'],
                hole=0.4,
                title="Daily Bias Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # === TAB 3: AI INSIGHTS ===
    with tab_ml:
        st.subheader("🤖 Pattern Recognition Engine (Beta)")
        st.markdown("""
        We trained a **Random Forest Classifier** on your historical data.
        *Question:* "Will the High of the Day form in the NY AM Session (9am - 11am)?"
        """)
        
        if st.button("Train AI Model"):
            model, accuracy = train_hod_lod_model(stats_df)
            
            if model:
                st.success(f"Model Trained Successfully! Accuracy: **{accuracy:.1%}**")
                
                # Prediction for tomorrow
                if not stats_df.empty:
                    last_range = stats_df.iloc[-1]['Daily_Range']
                    day_code = (datetime.now().weekday() + 1) % 7 
                    
                    prediction = model.predict([[day_code, last_range]])[0]
                    prob = model.predict_proba([[day_code, last_range]])[0]
                    
                    st.markdown("### 🔮 Tomorrow's Prediction")
                    if prediction == 1:
                        st.success(f"**High Probability of AM High:** The AI predicts a **{prob[1]:.1f}%** chance the HOD will form between 9-11 AM tomorrow.")
                    else:
                        st.warning(f"**Low Probability of AM High:** The AI predicts the High will likely form outside the NY AM session ({prob[0]:.1f}% chance).")
            else:
                st.error("Not enough data to train model yet.")
        else:
            st.info("Click the button above to train the model on current data.")

    # === TAB 4: TOOLS & EXPORT ===
    with tab_tools:
        st.subheader("💻 Pine Script Generator")
        st.markdown("Copy this code into TradingView.")
        
        best_h_hour = stats_df['HOD_Hour'].mode()[0]
        best_l_hour = stats_df['LOD_Hour'].mode()[0]
        
        pine_script = f"""
        // Generated by Streamlit App
        //@version=5
        indicator("Statistical High/Low Time", overlay=true)
        
        // Best Hours
        highTime = {best_h_hour}
        lowTime = {best_l_hour}
        
        // Highlights
        isHighTime = hour == highTime
        isLowTime = hour == lowTime
        
        bgcolor(isHighTime ? color.new(color.red, 80) : na, title="High Prob Zone")
        bgcolor(isLowTime ? color.new(color.green, 80) : na, title="Low Prob Zone")
        """
        st.code(pine_script, language="pine")
        
        st.divider()
        st.download_button(
            label="Download CSV",
            data=stats_df.to_csv(index=False).encode('utf-8'),
            file_name=f'{selected_asset_name}_stats.csv',
            mime='text/csv',
        )

else:
    # If we are here, stats_df is None or empty even after the fallback check in spinner
    st.warning("⚠️ No data available to display.")
    st.markdown("""
    **Possible reasons:**
    1. The market is currently closed or the API is rate-limiting requests.
    2. The 'Days of History' is too short to find valid days.
    3. Try switching the Asset to 'S&P 500' to test if the connection works.
    """)

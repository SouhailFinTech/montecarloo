import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Monte Carlo Stock Forecast", layout="wide")

# Custom CSS for elite look
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight:bold;
}
.small-font {
    font-size:16px;
    color:#888;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="big-font">📈 MonteCarloCo – Stock Price Forecast</p>', unsafe_allow_html=True)
st.markdown('<p class="small-font">Built with Python | souhail78@gmail.com</p>', unsafe_allow_html=True)

# Sidebar Inputs
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Enter Ticker Symbol", "AAPL")
    days = st.slider("Days to Simulate", min_value=10, max_value=500, value=252)
    simulations = st.slider("Number of Simulations", min_value=10, max_value=1000, value=100)
    run_button = st.button("Run Simulation")

# Function to fetch data from Alpha Vantage
def get_alpha_vantage_data(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        st.error("🚫 Error fetching data – check ticker or API key")
        return None

    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. adjusted close": "Adj Close"
    })

    df = df[['Adj Close']]
    df = df.apply(pd.to_numeric)
    df = df.sort_index()
    df['Adj Close'] = df['Adj Close'].astype(float)
    
    return df

# Main App Logic 
if run_button:
    # Use your Alpha Vantage API key here
    API_KEY = "UYN4QP9SRWLKVPK9"

    st.info(f"📊 Fetching '{ticker}' data from Alpha Vantage...")
    df = get_alpha_vantage_data(ticker.upper(), API_KEY)

    if df is not None and not df.empty:
        data = df['Adj Close'].copy().dropna()

        log_returns = np.log(1 + data.pct_change().dropna())
        u = log_returns.mean()
        var = log_returns.var()
        stdev = log_returns.std()
        drift = u - 0.5 * var

        daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(days, simulations)))

        price_list = np.zeros_like(daily_returns)
        price_list[0] = float(data.iloc[-1])

        for t in range(1, days):
            price_list[t] = price_list[t - 1] * daily_returns[t]

        final_prices = price_list[-1]
        mean_price = round(final_prices.mean(), 2)
        upper_bound = round(np.percentile(final_prices, 95), 2)
        lower_bound = round(np.percentile(final_prices, 5), 2)

        # Plotting with Plotly
        fig = go.Figure()

        for i in range(simulations):
            fig.add_trace(go.Scatter(x=np.arange(days), y=price_list[:, i],
                                   mode='lines', name=f"Path {i+1}",
                                   line=dict(color='navy'), opacity=0.3,
                                   showlegend=False))

        fig.add_trace(go.Scatter(x=np.arange(days), y=np.mean(price_list, axis=1),
                      mode='lines', name='Mean Path',
                      line=dict(color='gold', width=3)))

        fig.update_layout(title=f"{ticker.upper()} – {simulations} Simulated Paths",
                          xaxis_title="Day",
                          yaxis_title="Simulated Price ($)",
                          template="plotly_dark",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Price", f"${mean_price}")
        col2.metric("Upper Bound (95%)", f"${upper_bound}")
        col3.metric("Lower Bound (5%)", f"${lower_bound}")

        # Summary Text
        st.markdown(f"""
        ### 📊 Summary
        
        Based on the last 2 years of historical performance, we simulated **{simulations} possible future paths** over **{days} trading days**.

        - **Current Price**: ${round(data.iloc[-1], 2)}
        - **Expected Price after {days} days**: ${mean_price}
        - **95% Confidence Interval**: [{lower_bound}, {upper_bound}]
        - **Upside Target (95%)**: {(upper_bound / data.iloc[-1] - 1) * 100:.2f}% gain
        - **Downside Risk (5%)**: {(data.iloc[-1] / lower_bound - 1) * 100 - 100:.2f}% loss
        """)

        # Export Button
        df_export = pd.DataFrame(price_list)
        csv = df_export.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Simulated Paths (CSV)",
            data=csv,
            file_name=f"{ticker}_monte_carlo_simulation.csv",
            mime="text/csv"
        )

    else:
        st.warning("🚫 No valid data received – check your ticker symbol or API key")

else:
    st.info("📌 Enter a ticker symbol and click 'Run Simulation' to get started.")
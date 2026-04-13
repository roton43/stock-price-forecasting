import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Stock Analyzer & Forecaster", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #00C6FF;'>
      Stock Analyzer & Forecast System
    </h1>
""", unsafe_allow_html=True)

st.markdown("Analyze historical trends, technical indicators, and forecast stock prices.")

st.sidebar.title("Forecast Parameters")

stock_symbol = st . sidebar . selectbox (
" Select Stock " ,
options =[ " AAPL " , " GOOGL " , " MSFT " , " TSLA " , " AMZN " , " META " , " NFLX " , " NVDA " ] ,
format_func = lambda x : {
" AAPL " : " Apple ( AAPL ) " ,
" GOOGL " : " Google ( GOOGL ) " ,
" MSFT " : " Microsoft ( MSFT ) " ,
" TSLA " : " Tesla ( TSLA ) " ,
" AMZN " : " Amazon ( AMZN ) " ,
" META " : " Meta ( META ) " ,
" NFLX " : " Netflix ( NFLX ) " ,
" NVDA " : " NVIDIA ( NVDA ) "
}[ x ]
)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", datetime.now() - timedelta(days=365*2))
end_date = col2.date_input("End Date", datetime.now())

horizon = st.sidebar.slider("Forecast Horizon (Business Days)", 7, 180, 90)

@st.cache_data(show_spinner=False)
def load_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

data = load_data(stock_symbol, start_date, end_date)

if data is None or data.empty:
    st.error("No data found.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data.dropna(inplace=True)

def safe(x):
    return float(np.squeeze(x))


def mean_absolute_error(actual, predicted):
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    return np.mean(np.abs(actual_arr - predicted_arr))

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Technical Analysis", "Forecast", "Statistics"])

with tab1:
    st.subheader(f"{stock_symbol} - Overview")

    current_price = safe(data['Close'].iloc[-1])
    prev_price = safe(data['Close'].iloc[-2])
    delta = current_price - prev_price

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"${current_price:.2f}", f"{delta:.2f}")
    col2.metric("52W High", f"${safe(data['High'].max()):.2f}")
    col3.metric("52W Low", f"${safe(data['Low'].min()):.2f}")
    col4.metric("Avg Volume", f"{int(data['Volume'].mean()):,}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))
    fig.update_layout(title="Candlestick Chart", height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Technical Indicators")

    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close"))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA50'], name="MA50"))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA200'], name="MA200"))
    fig_ma.update_layout(title="Moving Averages", showlegend=True)
    st.plotly_chart(fig_ma, use_container_width=True)

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig_rsi.add_trace(go.Scatter(
        x=[data.index.min(), data.index.max()],
        y=[70, 70],
        mode="lines",
        name="Overbought (70)",
        line=dict(color="red", dash="dash")
    ))
    fig_rsi.add_trace(go.Scatter(
        x=[data.index.min(), data.index.max()],
        y=[30, 30],
        mode="lines",
        name="Oversold (30)",
        line=dict(color="green", dash="dash")
    ))
    fig_rsi.update_layout(title="RSI", showlegend=True)
    st.plotly_chart(fig_rsi, use_container_width=True)

    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['Signal'], name="Signal"))
    fig_macd.update_layout(title="MACD", showlegend=True)
    st.plotly_chart(fig_macd, use_container_width=True)

    data['BB_MA'] = data['Close'].rolling(20).mean()
    data['BB_Upper'] = data['BB_MA'] + 2 * data['Close'].rolling(20).std()
    data['BB_Lower'] = data['BB_MA'] - 2 * data['Close'].rolling(20).std()

    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close"))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_MA'], name="Middle Band"))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name="Upper Band"))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name="Lower Band"))
    fig_bb.update_layout(title="Bollinger Bands", showlegend=True)
    st.plotly_chart(fig_bb, use_container_width=True)

with tab3:
    st.subheader("Forecast with Prophet")

    df_prophet = data.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']

    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
    df_prophet.dropna(inplace=True)

    with st.spinner("Training model..."):
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative'
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=horizon, freq='B')
        forecast = model.predict(future)

    forecast_fig = model.plot(forecast)
    if forecast_fig.axes:
        forecast_fig.axes[0].legend(["Historical Data", "Forecast", "Confidence Interval"])
    st.pyplot(forecast_fig)

    components_fig = model.plot_components(forecast)
    for ax in components_fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend()
    st.pyplot(components_fig)

    st.subheader("Forecast Accuracy")
    actual = df_prophet['y'][-100:]
    predicted = forecast['yhat'][:len(actual)]

    mae = mean_absolute_error(actual, predicted)
    st.metric("Model MAE", f"{mae:.2f}")

    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30 + horizon)
    st.dataframe(forecast_display)

    st.download_button("Download Forecast CSV",
                       forecast_display.to_csv(index=False),
                       f"{stock_symbol}_forecast.csv")

with tab4:
    st.subheader("Statistics")

    st.dataframe(data.describe())

    returns = data['Close'].pct_change().dropna()

    cagr = (data['Close'].iloc[-1] / data['Close'].iloc[0]) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    max_dd = (data['Close'] / data['Close'].cummax() - 1).min()

    col1, col2, col3 = st.columns(3)

    col1.metric("CAGR", f"{cagr:.2%}")
    col2.metric("Volatility", f"{volatility:.2%}")
    col3.metric("Max Drawdown", f"{max_dd:.2%}")

    st.download_button("Download Data CSV",
                       data.to_csv(),
                       f"{stock_symbol}_data.csv")

st.caption("Prepared by Md. Roton Ahmed | Data Source: Yahoo Finance | Model: Prophet")

# Stock Price Analyzer & Forecast App

A powerful **Stock Analysis & Forecasting Web App** built with **Streamlit**, featuring technical indicators, interactive visualizations, and future price prediction using Prophet.

---

## Features

### Market Analysis

* Real-time stock data using yfinance
* Interactive candlestick charts
* Historical trend visualization

### Technical Indicators

* Moving Averages (50-day, 200-day)
* RSI (Relative Strength Index)
* MACD (Moving Average Convergence Divergence)
* Bollinger Bands

### Forecasting

* Time series forecasting with Prophet
* Business-day future predictions
* Forecast confidence intervals

### Statistics & Metrics

* CAGR (Compound Annual Growth Rate)
* Volatility
* Maximum Drawdown
* Model evaluation (MAE)

### User Experience

* Clean, responsive UI with Streamlit
* Sidebar controls for customization
* Downloadable CSV data

---

## Tech Stack

* Frontend/UI: Streamlit
* Data Handling: Pandas, NumPy
* Visualization: Plotly
* Forecasting: Prophet
* Data Source: yfinance

---

## Demo

**Live Link** - ""

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/roton43/stock-price-forecast.git
cd stock-price-forecast
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**

```bash
venv\\Scripts\\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the App

```bash
streamlit run app.py
```

### 6. Open in Browser

```
http://localhost:8501
```

---

## Project Structure

```
stock-price-forecast/
│
├── app.py
├── requirements.txt
├── README.md
└── assets/
```

---

## Important Notes

* Requires Python 3.9 – 3.11 (Prophet may not support newer versions)
* Internet connection required (for stock data)
* Forecasts are not financial advice

---

## Use Cases

* Academic projects (ML / Data Science)
* Portfolio showcase
* Stock market analysis
* Time-series forecasting learning

---

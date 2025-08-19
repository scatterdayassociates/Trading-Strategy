import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import traceback
import vectorbt as vbt
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Trading Strategy Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

    
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: sans-serif;
        color: #1f2937;
    }
    
    /* Dashboard Container */
    .main .block-container {
        padding: 40px 30px;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Main Dashboard Header */
    .main-header {
        text-align: center;
        font-size: 32px;
        font-weight: 600;
        margin-bottom: 40px;
        color: #1f2937;
        letter-spacing: -0.02em;
    }
    
    /* Enhanced Card Styles with Gradients */
    .metric-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            padding: 20px;
            border-left: 4px solid #10b981;
        }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card-with-tooltip {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
        border: none;
        position: relative;
        color: white;
    }
    
    .metric-card-with-tooltip:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }
    
    .metric-title {
        font-size: 28px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 16px;
        text-align: center;
        letter-spacing: -0.01em;
        line-height: 1.4;
    }

    .metric-value {
        font-size: 38px;
        font-weight: 700;
        margin: 12px 0;
        color: white;
        letter-spacing: -0.02em;
    }
    
    .metric-unit {
        font-size: 14px;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 400;
        letter-spacing: -0.01em;
    }
    
    /* Chart Card Styling */
    .stPlotlyChart {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: none;
    }
    
    .stPlotlyChart:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    /* Performance Metrics Cards */
    .performance-metric-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: none;
        color: white;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .performance-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.15);
    }
    
    .performance-metric-value {
        font-size: 24px;
        font-weight: 700;
        margin: 8px 0;
        color: white;
        letter-spacing: -0.02em;
    }
    
    .performance-metric-label {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        line-height: 1.3;
        letter-spacing: -0.01em;
    }
    
    /* Weight Distribution Cards */
    .weight-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .weight-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
    }
    
    .weight-card-gradient {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
    }
    
    .weight-card-gradient .weight-label {
        color: rgba(255, 255, 255, 0.9);
    }
    
    .weight-card-gradient .weight-value {
        color: #fbbf24;
    }
    
    .weight-card-gradient .weight-unit {
        color: rgba(255, 255, 255, 0.7);
    }
    
    .weight-label {
        font-size: 14px;
        font-weight: 600;
        color: #495057;
        margin-bottom: 8px;
    }
    
    .weight-value {
        font-size: 24px;
        font-weight: 700;
        color: #007bff;
        margin: 4px 0;
    }
    
    .weight-unit {
        font-size: 12px;
        color: #6c757d;
        font-weight: 400;
    }
    
    /* Tooltip Enhancements */
    .tooltip-container {
        position: relative;
        display: inline-block;
    }
    
    .tooltip-icon {
        font-size: 14px;
        margin-left: 6px;
        color: #0891b2;
        cursor: help;
        transition: color 0.2s ease;
    }
    
    .tooltip-icon:hover {
        color: #0c7489;
    }
    
    .tooltip-text {
        visibility: hidden;
        width: 280px;
        background: #1f2937;
        color: #ffffff;
        text-align: center;
        border-radius: 8px;
        padding: 12px;
        position: absolute;
        z-index: 1000;
        bottom: 130%;
        left: 50%;
        margin-left: -140px;
        opacity: 0;
        transition: all 0.3s ease;
        font-size: 12px;
        line-height: 1.4;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        font-weight: 400;
    }
    
    .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -6px;
        border-width: 6px;
        border-style: solid;
        border-color: #1f2937 transparent transparent transparent;
    }
    
    .tooltip-container:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    
    /* Delta Overlay Enhancements */
    .delta-overlay {
        margin-top: 12px;
        display: flex;
        flex-direction: column;
        gap: 4px;
        background: #f8fafc;
        padding: 6px 8px;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }

    .delta-item {
        display: flex;
        align-items: baseline;
        gap: 4px;
        font-size: 11px;
        white-space: nowrap;
        justify-content: center;
    }

    .delta-label {
        font-size: 11px;
        font-weight: 600;
        color: #64748b;
    }

    .delta-value {
        font-size: 12px;
        font-weight: 700;
    }

    .delta-unit {
        font-size: 10px;
        color: #64748b;
        font-weight: 400;
    }
        
    /* Enhanced Expander Styles */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 20px 24px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #1f2937 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
    }
    
    .streamlit-expanderContent {
        background: #ffffff !important;
        border-radius: 0 0 12px 12px !important;
        padding: 24px !important;
        border: none !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1) !important;
        margin-bottom: 24px !important;
    }
    
    .streamlit-expander {
        border: none !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1) !important;
        border-radius: 12px !important;
        margin: 24px 0 !important;
        background: #ffffff !important;
    }
    
    /* Accuracy Card Enhancements */
    .accuracy-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border: none;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .accuracy-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
    }
    
    .accuracy-value {
        font-size: 22px;
        font-weight: 700;
        margin: 8px 0;
        letter-spacing: -0.02em;
    }
    
    .accuracy-label {
        font-size: 12px;
        color: #64748b;
        font-weight: 500;
        line-height: 1.3;
        letter-spacing: -0.01em;
    }
    
    /* PLUTO Section Enhancements */
    .pluto-container {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 28px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .pluto-item {
        font-size: 15px;
        color: #1f2937;
        margin-bottom: 12px;
        line-height: 1.5;
    }
    
    .pluto-label {
        font-weight: 600;
        display: inline;
        color: #374151;
    }
    
    .pluto-value {
        font-weight: 400;
        display: inline;
        color: #1f2937;
    }
    
    .pluto-section-header {
        font-size: 20px;
        font-weight: 600;
        color: #1f2937;
        margin: 40px 0 24px 0;
        border-bottom: 2px solid #0891b2;
        padding-bottom: 12px;
        letter-spacing: -0.01em;
    }
    
        /* Enhanced Sidebar Styling */
    .css-1d391kg {
        background: #f8fafc;
        padding: 0;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
    }
    
        /* Sidebar Section Headers */
    .sidebar-header {
        font-size: 16px;
        font-weight: 500;
        color: #374151;
        margin: 24px 0 16px 0;
        padding: 0;
    }
    
        /* Sidebar Cards */
    .sidebar-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 0;
        margin: 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }
    
    /* Button Enhancements */
    .stButton > button {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        background: #2563eb !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Override Streamlit's primary button styling */
    .stButton > button[data-baseweb="button"] {
        background: #3b82f6 !important;
        color: white !important;
    }
    
    /* Force blue color for all button states */
    button[data-testid="baseButton-primary"] {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
    }
    
    /* Additional Streamlit button overrides */
    [data-testid="baseButton-primary"] {
        background: #3b82f6 !important;
        color: white !important;
    }
    
    /* Override any Streamlit primary button */
    .stButton > button[type="primary"] {
        background: #3b82f6 !important;
        color: white !important;
    }
    

    
    /* Spacing improvements */
    .element-container {
        margin-bottom: 0;
    }
    
    /* Remove spacing from sidebar elements */
    .css-1d391kg .element-container {
        margin: 0;
        padding: 0;
    }
    
    .css-1d391kg .stMarkdown {
        margin: 0;
        padding: 0;
    }
    
    .css-1d391kg .stTextInput, 
    .css-1d391kg .stTextArea,
    .css-1d391kg .stSlider,
    .css-1d391kg .stCheckbox,
    .css-1d391kg .stDateInput,
    .css-1d391kg .stButton {
        margin: 0;
        padding: 0;
    }
    
    /* Grid layout improvements */
    .row-widget {
        gap: 20px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #1f2937;
        margin: 40px 0 24px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* H3 elements with section-header class */
    h3.section-header {
        font-size: 18px;
        font-weight: 600;
        color: #1f2937;
        margin: 40px 0 24px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* Inline H3 elements in chart components */
    h3[style*="font-size: 18px"] {
        margin: 40px 0 24px 0 !important;
        padding-bottom: 12px !important;
        border-bottom: 1px solid #e5e7eb !important;
    }
    
    /* Main page title spacing */
    h1[style*="font-size: 24px"] {
        margin: 0 0 32px 0 !important;
        padding: 0 !important;
    }
    
    /* Ensure consistent spacing for all section headers */
    .section-header, h2.section-header, h3.section-header {
        margin: 40px 0 24px 0 !important;
        padding-bottom: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# Set vectorbt theme
vbt.settings.set_theme("dark")

# Helper functions
def get_last_valid(series):
    if series is None or not isinstance(series, pd.Series) or series.dropna().empty:
        return np.nan
    return series.dropna().iloc[-1]

def calculate_bb_pos_series(close_series, bb_lower_series, bb_upper_series):
    if not isinstance(close_series, pd.Series) or not isinstance(bb_lower_series, pd.Series) or not isinstance(bb_upper_series, pd.Series):
        return pd.Series(np.nan, index=close_series.index if close_series is not None else None)
    bb_range = bb_upper_series - bb_lower_series
    bb_pos = np.where(bb_range <= 0, 0.5, (close_series - bb_lower_series) / bb_range)
    bb_pos = pd.Series(bb_pos, index=close_series.index)
    return bb_pos.clip(-1, 2)

def lorentzian_distance_daily(series):
    returns_shifted = series.shift(1)
    squared_diff = (series - returns_shifted)**2
    distance = np.log1p(squared_diff)
    return pd.Series(distance, index=series.index)

def parse_tickers(ticker_input):
    """Parse ticker input string into list of tickers"""
    if not ticker_input:
        return []
    
    # Split by comma, semicolon, or space and clean up
    tickers = []
    for separator in [',', ';', ' ']:
        if separator in ticker_input:
            tickers = [t.strip().upper() for t in ticker_input.split(separator)]
            break
    else:
        # No separator found, treat as single ticker
        tickers = [ticker_input.strip().upper()]
    
    # Remove empty strings and duplicates while preserving order
    seen = set()
    clean_tickers = []
    for ticker in tickers:
        if ticker and ticker not in seen:
            clean_tickers.append(ticker)
            seen.add(ticker)
    
    return clean_tickers

class StockAnalyzer:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.processed_data = {}
        # Initialize with default weights that sum to 1
        self.scoring_rules = {
            'predicted_return': {
                'positive': 1, 'negative': -1, 'weight': 0.4,
                'description': "Predicted Return"
            },
            'confidence_interval': {
                'both_positive': 1, 'both_negative': -1, 'mixed': 0, 'weight': 0.3,
                'description': "Confidence Interval"
            },
            'rsi': {
                'range': (30, 70), 'within_range': 0, 'overbought': -1, 'oversold': 0,
                'weight': 0.1, 'description': "RSI(14)"
            },
            'bb_position': {
                'range': (0.2, 0.8), 'within_range': 0, 'above_range': -1, 'below_range': 1,
                'weight': 0.1, 'description': "Bollinger %"
            },
            'sma_cross': {
                'sma20_above_sma50': 1, 'sma20_below_sma50': -1, 'weight': 0.1,
                'description': "SMA Cross"
            }
        }

    def get_clean_financial_data(self, ticker_symbol):
        try:
            data_df = yf.download(ticker_symbol,
                                start=self.start_date,
                                end=self.end_date,
                                auto_adjust=True,
                                progress=False)

            if data_df.empty:
                st.warning(f"No data downloaded for {ticker_symbol}")
                return None

            if isinstance(data_df.columns, pd.MultiIndex):
                if len(data_df.columns.levels) > 1:
                    try:
                        second_level_names = data_df.columns.get_level_values(1)
                        if all(name in [ticker_symbol, '', None] for name in second_level_names):
                            data_df.columns = data_df.columns.droplevel(1)
                        else:
                            raise ValueError("Unexpected MultiIndex second level names.")
                    except Exception:
                        data_df.columns = ['_'.join(map(str, col)).strip() for col in data_df.columns.values]
                        data_df.columns = data_df.columns.str.rstrip('_')

            found_close = False
            for col_name in data_df.columns:
                if 'close' in col_name.lower():
                    if col_name != 'Close':
                        data_df = data_df.rename(columns={col_name: 'Close'})
                    found_close = True
                    break
            if not found_close:
                st.error(f"Could not identify a 'Close' column for {ticker_symbol}")
                return None

            if 'Close' not in data_df.columns:
                found_close = False
                for col_name in data_df.columns:
                    if 'close' in col_name.lower():
                        if col_name != 'Close':
                            data_df = data_df.rename(columns={col_name: 'Close'})
                        found_close = True
                        break
                if not found_close:
                    st.error(f"'Close' column not found for {ticker_symbol}")
                    return None

            close_prices_series = data_df['Close']
            if not isinstance(close_prices_series, pd.Series) or not pd.api.types.is_numeric_dtype(close_prices_series):
                st.error(f"'Close' column for {ticker_symbol} is not numeric")
                return None

            data_df = data_df.ffill()
            data_df = data_df.dropna(subset=['Close'])

            if data_df.empty or data_df['Close'].empty:
                st.warning(f"Data for {ticker_symbol} is empty after cleaning")
                return None

            if data_df.index.tz is not None:
                data_df.index = data_df.index.tz_localize(None)

            essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            cols_to_keep = [col for col in essential_cols if col in data_df.columns]
            if 'Close' not in cols_to_keep:
                st.error(f"'Close' not in cols_to_keep for {ticker_symbol}")
                return None

            final_df = data_df[list(set(cols_to_keep))].copy()
            return final_df

        except Exception as e:
            st.error(f"Error in get_clean_financial_data for {ticker_symbol}: {str(e)}")
            return None

    @staticmethod
    def calculate_technical_indicators(data):
        if data is None or 'Close' not in data.columns or data['Close'].empty:
            st.warning("Invalid data passed to TI calc")
            if data is not None: return data.copy()
            return pd.DataFrame(index=data.index if data is not None else None)

        data_ti = data.copy()
        close_prices_s = data_ti['Close']
        if not isinstance(close_prices_s, pd.Series):
            st.error("data_ti['Close'] is not a pd.Series")
            return data_ti

        data_ti['SMA_20'] = close_prices_s.rolling(window=20, min_periods=1).mean()
        data_ti['SMA_50'] = close_prices_s.rolling(window=50, min_periods=1).mean()

        delta = close_prices_s.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=1).mean()
        rs = np.divide(gain, loss, out=np.full_like(gain, np.inf), where=(loss != 0))
        rs = np.where((gain > 0) & (loss == 0), np.inf, rs)
        data_ti['RSI'] = 100 - (100 / (1 + rs))
        data_ti['RSI'] = data_ti['RSI'].fillna(50)

        bb_middle_s = close_prices_s.rolling(window=20, min_periods=1).mean()
        std_dev_s = close_prices_s.rolling(window=20, min_periods=1).std(ddof=0).fillna(0)
        data_ti['BB_middle'] = bb_middle_s
        data_ti['BB_upper'] = bb_middle_s + (2 * std_dev_s)
        data_ti['BB_lower'] = bb_middle_s - (2 * std_dev_s)
        data_ti['BB_Position'] = calculate_bb_pos_series(data_ti['Close'], data_ti['BB_lower'], data_ti['BB_upper'])

        return data_ti

    def calculate_rolling_prediction_and_anomaly(self, data_df):
        if data_df is None or data_df.empty or 'Daily_Return' not in data_df.columns:
            st.warning("Invalid data for rolling prediction/anomaly calculation.")
            return data_df if data_df is not None else pd.DataFrame()

        data_calc = data_df.copy()
        returns_s = data_calc['Daily_Return']

        data_calc['Lorentzian_Distance'] = lorentzian_distance_daily(returns_s)

        threshold_window = 60
        min_periods_threshold = 30
        rolling_mean_dist = data_calc['Lorentzian_Distance'].rolling(window=threshold_window, min_periods=min_periods_threshold).mean()
        rolling_std_dist = data_calc['Lorentzian_Distance'].rolling(window=threshold_window, min_periods=min_periods_threshold).std()
        rolling_mean_dist_shifted = rolling_mean_dist.shift(1)
        rolling_std_dist_shifted = rolling_std_dist.shift(1)
        data_calc['Lorentzian_Threshold'] = (rolling_mean_dist_shifted + 2 * rolling_std_dist_shifted).fillna(np.inf)
        data_calc['Anomaly_Detected'] = (data_calc['Lorentzian_Distance'] > data_calc['Lorentzian_Threshold']) & (data_calc['Lorentzian_Distance'].notna())

        pred_mean_window = 10
        pred_std_window = 30
        data_calc['Predicted_Return'] = returns_s.rolling(window=pred_mean_window, min_periods=1).mean()
        rolling_std_pred = returns_s.rolling(window=pred_std_window, min_periods=2).std()
        rolling_count_pred_std = returns_s.rolling(window=pred_std_window, min_periods=2).count()
        std_err_pred = np.divide(rolling_std_pred, np.sqrt(rolling_count_pred_std), out=np.full_like(rolling_std_pred, np.nan), where=(rolling_count_pred_std > 0))

        z_score_95 = stats.norm.ppf((1 + 0.95) / 2)
        data_calc['Predicted_CI_Low'] = data_calc['Predicted_Return'] - z_score_95 * std_err_pred
        data_calc['Predicted_CI_High'] = data_calc['Predicted_Return'] + z_score_95 * std_err_pred

        return data_calc

    @staticmethod
    def calculate_daily_scores(data_df, scoring_rules):
        if data_df is None or data_df.empty:
            return pd.DataFrame(columns=list(scoring_rules.keys()) + ['Total_Score']), pd.DataFrame(columns=list(scoring_rules.keys()))

        scores_df = pd.DataFrame(index=data_df.index)

        # Predicted Return Score
        pred_return = data_df.get('Predicted_Return', pd.Series(np.nan, index=data_df.index))
        pred_ret_score = np.where(pred_return > 0,
                                   scoring_rules['predicted_return']['positive'],
                                   scoring_rules['predicted_return']['negative'])
        anomaly_col = data_df.get('Anomaly_Detected', pd.Series(False, index=data_df.index)).fillna(False)
        zero_pred_score_cond = pred_return.isna() | anomaly_col
        pred_ret_score = np.where(zero_pred_score_cond, 0, pred_ret_score)
        scores_df['predicted_return'] = pred_ret_score

        # Confidence Interval Score
        conf_low = data_df.get('Predicted_CI_Low', pd.Series(np.nan, index=data_df.index))
        conf_high = data_df.get('Predicted_CI_High', pd.Series(np.nan, index=data_df.index))
        
        ci_score = np.select(
            [ (conf_low > 0) & (conf_high > 0), (conf_low < 0) & (conf_high < 0) ],
            [ scoring_rules['confidence_interval']['both_positive'],
              scoring_rules['confidence_interval']['both_negative'] ],
            default=scoring_rules['confidence_interval']['mixed']
        )
        zero_ci_score_cond = conf_low.isna() | conf_high.isna() | anomaly_col
        ci_score = np.where(zero_ci_score_cond, 0, ci_score)
        scores_df['confidence_interval'] = ci_score

        # RSI Score
        rsi = data_df.get('RSI', pd.Series(np.nan, index=data_df.index))
        rsi_low, rsi_high = scoring_rules['rsi']['range']
        rsi_score = np.select(
            [ rsi < rsi_low, rsi > rsi_high ],
            [ scoring_rules['rsi']['oversold'], scoring_rules['rsi']['overbought'] ],
            default=scoring_rules['rsi']['within_range']
        )
        rsi_score = np.where(rsi.isna(), 0, rsi_score)
        scores_df['rsi'] = rsi_score

        # Bollinger Bands Position Score
        bb_pos = data_df.get('BB_Position', pd.Series(np.nan, index=data_df.index))
        bb_low, bb_high = scoring_rules['bb_position']['range']
        bb_pos_score = np.select(
            [ bb_pos < bb_low, bb_pos > bb_high ],
            [ scoring_rules['bb_position']['below_range'], scoring_rules['bb_position']['above_range'] ],
            default=scoring_rules['bb_position']['within_range']
        )
        bb_pos_score = np.where(bb_pos.isna(), 0, bb_pos_score)
        scores_df['bb_position'] = bb_pos_score

        # SMA Cross Score
        sma20 = data_df.get('SMA_20', pd.Series(np.nan, index=data_df.index))
        sma50 = data_df.get('SMA_50', pd.Series(np.nan, index=data_df.index))

        sma_cross_score = np.select(
            [ sma20 > sma50, sma20 < sma50 ],
            [ scoring_rules['sma_cross']['sma20_above_sma50'], scoring_rules['sma_cross']['sma20_below_sma50'] ],
            default=0
        )
        sma_cross_score = np.where(sma20.isna() | sma50.isna(), 0, sma_cross_score)
        scores_df['sma_cross'] = sma_cross_score

        # Calculate Total Score with proper weight application
        total_score = pd.Series(0.0, index=data_df.index)
        for indicator_name, rule in scoring_rules.items():
            if indicator_name in scores_df.columns:
                daily_scores = scores_df[indicator_name].fillna(0)
                total_score += daily_scores * rule['weight']

        # Ensure total score is finite and within reasonable bounds
        total_score = total_score.fillna(0)
        total_score = np.clip(total_score, -2, 2)  # Clip to reasonable range

        data_df_with_scores = data_df.copy()
        data_df_with_scores['Total_Score'] = total_score

        return data_df_with_scores, scores_df

    def analyze_stocks(self):
        self.processed_data = {}
        failed_tickers = []

        for ticker in self.tickers:
            data_with_ohlc = self.get_clean_financial_data(ticker)

            if data_with_ohlc is None or data_with_ohlc.empty or \
               'Close' not in data_with_ohlc.columns or data_with_ohlc['Close'].empty or \
               not isinstance(data_with_ohlc['Close'], pd.Series):
                st.error(f"Critical failure: Could not process valid 'Close' Series for {ticker}.")
                failed_tickers.append(ticker)
                continue

            data_with_ohlc['Daily_Return'] = data_with_ohlc['Close'].pct_change()
            data_with_ti = self.calculate_technical_indicators(data_with_ohlc)
            data_with_pred = self.calculate_rolling_prediction_and_anomaly(data_with_ti)

            if data_with_pred is None or data_with_pred.empty:
                st.error(f"Processing resulted in empty data after prediction calc for {ticker}.")
                failed_tickers.append(ticker)
                continue

            data_final, _ = self.calculate_daily_scores(data_with_pred, self.scoring_rules)

            if data_final is None or data_final.empty:
                st.error(f"Processing resulted in empty data after score calc for {ticker}.")
                failed_tickers.append(ticker)
                continue

            self.processed_data[ticker] = data_final

        # Update tickers list to exclude failed ones
        self.tickers = [t for t in self.tickers if t not in failed_tickers]
        
        if failed_tickers:
            st.warning(f"Failed to process the following tickers: {', '.join(failed_tickers)}")

class VectorbtTradingStrategy:
    def __init__(self, processed_data, decision_thresholds, scoring_rules, take_profit_pct, stop_loss_pct):
        self.processed_data = processed_data
        self.decision_thresholds = decision_thresholds
        self.scoring_rules = scoring_rules
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.transaction_log = []
        self.portfolio_history = []
        self.cash_balance = 100000  # Initial capital
        self.positions = {}  # Track positions for each ticker
        self.fee_rate = 0.001  # 0.1% transaction fee
        self.trade_count = 0
        self.first_trade_date = None
        self.last_trade_date = None
        self.initial_capital = 100000

    def calculate_portfolio_value(self, current_date):
        """Calculate current portfolio value including cash and positions"""
        total_position_value = 0
        for ticker, pos in self.positions.items():
            if pos['shares'] > 0 and ticker in self.processed_data:
                if current_date in self.processed_data[ticker].index:
                    current_price = self.processed_data[ticker].loc[current_date, 'Close']
                    total_position_value += pos['shares'] * current_price
        return self.cash_balance + total_position_value

    def generate_vbt_signals(self):
        all_entries = {}
        all_exits = {}
        price_data = {}

        for ticker, data_df in self.processed_data.items():
            if data_df.empty or 'Close' not in data_df.columns:
                continue

            close_prices = data_df['Close']
            entries = pd.Series(False, index=data_df.index)
            exits = pd.Series(False, index=data_df.index)

            # Initialize position tracking for this ticker
            self.positions[ticker] = {'shares': 0, 'entry_price': 0, 'entry_date': None}

            for i in range(len(data_df)):
                current_date = data_df.index[i]
                current_price = close_prices.iloc[i]

                # Exit logic first (check if we should sell existing positions)
                if self.positions[ticker]['shares'] > 0:  # In position for this ticker
                    entry_price = self.positions[ticker]['entry_price']
                    shares = self.positions[ticker]['shares']
                    returns = (current_price - entry_price) / entry_price

                    # Take profit
                    if returns >= self.take_profit_pct / 100:
                        exits.iloc[i] = True
                        proceeds = shares * current_price
                        fee = proceeds * self.fee_rate
                        net_proceeds = proceeds - fee
                        self.cash_balance += net_proceeds
                        self.last_trade_date = current_date

                        # Update portfolio value after transaction
                        updated_portfolio_value = self.calculate_portfolio_value(current_date)

                        # Log transaction
                        self.transaction_log.append({
                            'date': current_date,
                            'ticker': ticker,
                            'type': 'SELL-TP',
                            'price': current_price,
                            'shares': shares,
                            'amount': proceeds,
                            'fee': fee,
                            'cash_balance': self.cash_balance,
                            'portfolio_value': updated_portfolio_value,
                            'profit_pct': returns * 100
                        })
                        self.positions[ticker] = {'shares': 0, 'entry_price': 0, 'entry_date': None}
                        self.trade_count += 1

                    # Stop loss
                    elif returns <= -self.stop_loss_pct / 100:
                        exits.iloc[i] = True
                        proceeds = shares * current_price
                        fee = proceeds * self.fee_rate
                        net_proceeds = proceeds - fee
                        self.cash_balance += net_proceeds
                        self.last_trade_date = current_date

                        # Update portfolio value after transaction
                        updated_portfolio_value = self.calculate_portfolio_value(current_date)

                        # Log transaction
                        self.transaction_log.append({
                            'date': current_date,
                            'ticker': ticker,
                            'type': 'SELL-SL',
                            'price': current_price,
                            'shares': shares,
                            'amount': proceeds,
                            'fee': fee,
                            'cash_balance': self.cash_balance,
                            'portfolio_value': updated_portfolio_value,
                            'profit_pct': returns * 100
                        })
                        self.positions[ticker] = {'shares': 0, 'entry_price': 0, 'entry_date': None}
                        self.trade_count += 1

                # Entry logic (check if we should buy new positions)
                if self.positions[ticker]['shares'] == 0:  # Not in position for this ticker
                    score = data_df['Total_Score'].iloc[i]
                    if score >= self.decision_thresholds['buy']:
                        # Calculate position size (use available cash divided by number of tickers)
                        available_cash = self.cash_balance / len(self.processed_data)
                        shares = int(available_cash / current_price)
                        if shares == 0:  # Not enough cash
                            continue

                        cost = shares * current_price
                        fee = cost * self.fee_rate
                        total_cost = cost + fee

                        if total_cost > self.cash_balance:
                            continue

                        # Execute buy
                        entries.iloc[i] = True
                        self.positions[ticker] = {
                            'shares': shares,
                            'entry_price': current_price,
                            'entry_date': current_date
                        }
                        self.cash_balance -= total_cost
                        self.trade_count += 1

                        # Update first trade date if needed
                        if self.first_trade_date is None or current_date < self.first_trade_date:
                            self.first_trade_date = current_date

                        # Update portfolio value after transaction
                        updated_portfolio_value = self.calculate_portfolio_value(current_date)

                        # Log transaction
                        self.transaction_log.append({
                            'date': current_date,
                            'ticker': ticker,
                            'type': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'amount': cost,
                            'fee': fee,
                            'cash_balance': self.cash_balance,
                            'portfolio_value': updated_portfolio_value
                        })

                # Update portfolio history after all transactions for this date
                current_portfolio_value = self.calculate_portfolio_value(current_date)
                if not self.portfolio_history or self.portfolio_history[-1]['date'] != current_date:
                    self.portfolio_history.append({
                        'date': current_date,
                        'portfolio_value': current_portfolio_value,
                        'cash': self.cash_balance,
                        'position_value': current_portfolio_value - self.cash_balance
                    })

                # Entry logic
                if self.positions[ticker]['shares'] == 0:  # Not in position for this ticker
                    score = data_df['Total_Score'].iloc[i]
                    if score >= self.decision_thresholds['buy']:
                        # Calculate position size (use available cash divided by number of tickers)
                        available_cash = self.cash_balance / len(self.processed_data)
                        shares = int(available_cash / current_price)
                        if shares == 0:  # Not enough cash
                            continue

                        cost = shares * current_price
                        fee = cost * self.fee_rate
                        total_cost = cost + fee

                        if total_cost > self.cash_balance:
                            continue

                        # Execute buy
                        entries.iloc[i] = True
                        self.positions[ticker] = {
                            'shares': shares,
                            'entry_price': current_price,
                            'entry_date': current_date
                        }
                        self.cash_balance -= total_cost
                        self.trade_count += 1

                        # Update first trade date if needed
                        if self.first_trade_date is None or current_date < self.first_trade_date:
                            self.first_trade_date = current_date

                        # Update portfolio value after transaction
                        updated_portfolio_value = self.calculate_portfolio_value(current_date)

                        # Log transaction
                        self.transaction_log.append({
                            'date': current_date,
                            'ticker': ticker,
                            'type': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'amount': cost,
                            'fee': fee,
                            'cash_balance': self.cash_balance,
                            'portfolio_value': updated_portfolio_value
                        })

                # Exit logic
                if self.positions[ticker]['shares'] > 0:  # In position for this ticker
                    entry_price = self.positions[ticker]['entry_price']
                    shares = self.positions[ticker]['shares']
                    returns = (current_price - entry_price) / entry_price

                    # Take profit
                    if returns >= self.take_profit_pct / 100:
                        exits.iloc[i] = True
                        proceeds = shares * current_price
                        fee = proceeds * self.fee_rate
                        net_proceeds = proceeds - fee
                        self.cash_balance += net_proceeds
                        self.last_trade_date = current_date

                        # Update portfolio value after transaction
                        updated_portfolio_value = self.calculate_portfolio_value(current_date)

                        # Log transaction
                        self.transaction_log.append({
                            'date': current_date,
                            'ticker': ticker,
                            'type': 'SELL-TP',
                            'price': current_price,
                            'shares': shares,
                            'amount': proceeds,
                            'fee': fee,
                            'cash_balance': self.cash_balance,
                            'portfolio_value': updated_portfolio_value,
                            'profit_pct': returns * 100
                        })
                        self.positions[ticker] = {'shares': 0, 'entry_price': 0, 'entry_date': None}
                        self.trade_count += 1

                    # Stop loss
                    elif returns <= -self.stop_loss_pct / 100:
                        exits.iloc[i] = True
                        proceeds = shares * current_price
                        fee = proceeds * self.fee_rate
                        net_proceeds = proceeds - fee
                        self.cash_balance += net_proceeds
                        self.last_trade_date = current_date

                        # Update portfolio value after transaction
                        updated_portfolio_value = self.calculate_portfolio_value(current_date)

                        # Log transaction
                        self.transaction_log.append({
                            'date': current_date,
                            'ticker': ticker,
                            'type': 'SELL-SL',
                            'price': current_price,
                            'shares': shares,
                            'amount': proceeds,
                            'fee': fee,
                            'cash_balance': self.cash_balance,
                            'portfolio_value': updated_portfolio_value,
                            'profit_pct': returns * 100
                        })
                        self.positions[ticker] = {'shares': 0, 'entry_price': 0, 'entry_date': None}
                        self.trade_count += 1

            all_entries[ticker] = entries
            all_exits[ticker] = exits
            price_data[ticker] = close_prices

        # Filter price data to only include the active trading period
        if self.first_trade_date is not None and self.last_trade_date is not None:
            filtered_price_data = {}
            for ticker, prices in price_data.items():
                filtered_prices = prices.loc[self.first_trade_date:self.last_trade_date]
                filtered_price_data[ticker] = filtered_prices
            price_data = filtered_price_data

            # Also filter entries and exits
            filtered_entries = {}
            filtered_exits = {}
            for ticker in all_entries.keys():
                filtered_entries[ticker] = all_entries[ticker].loc[self.first_trade_date:self.last_trade_date]
                filtered_exits[ticker] = all_exits[ticker].loc[self.first_trade_date:self.last_trade_date]
            all_entries = filtered_entries
            all_exits = filtered_exits

        return pd.DataFrame(price_data), pd.DataFrame(all_entries), pd.DataFrame(all_exits)

class StrategyOptimizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.best_result = None
        self.results = []

    def generate_valid_weight_combinations(self, step=0.1):
        """Generate weight combinations that sum to 1"""
        weights = np.arange(0, 1 + step, step)
        valid_combinations = []

        # Generate combinations for 4 indicators (5th is 1-sum)
        for w1 in weights:
            for w2 in weights:
                for w3 in weights:
                    for w4 in weights:
                        remaining = 1 - (w1 + w2 + w3 + w4)
                        if remaining >= 0:  # Only keep valid combinations
                            valid_combinations.append((w1, w2, w3, w4, remaining))

        return valid_combinations

    def optimize_weights(self, max_combinations=50, take_profit_pct=10, stop_loss_pct=5, buy_threshold=0.6):
        """Test different weight combinations that sum to 1"""
        weight_combinations = self.generate_valid_weight_combinations(step=0.2)

        # Sample combinations if too many
        if len(weight_combinations) > max_combinations:
            weight_combinations = [weight_combinations[i] for i in
                                 sorted(np.random.choice(len(weight_combinations), max_combinations, replace=False))]

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, weights in enumerate(weight_combinations):
            progress_bar.progress((i + 1) / len(weight_combinations))
            status_text.text(f"Testing combination {i + 1}/{len(weight_combinations)}")
            
            # Normalize to ensure exact sum of 1
            weights = np.array(weights)
            weights = weights / weights.sum()

            # Update weights (order: predicted_return, confidence, rsi, bb, sma)
            self.analyzer.scoring_rules['predicted_return']['weight'] = weights[0]
            self.analyzer.scoring_rules['confidence_interval']['weight'] = weights[1]
            self.analyzer.scoring_rules['rsi']['weight'] = weights[2]
            self.analyzer.scoring_rules['bb_position']['weight'] = weights[3]
            self.analyzer.scoring_rules['sma_cross']['weight'] = weights[4]

            # Re-analyze and backtest using the same strategy as final analysis
            self.analyzer.analyze_stocks()
            strategy = VectorbtTradingStrategy(
                self.analyzer.processed_data,
                {'buy': buy_threshold},  # Use the user's buy threshold
                self.analyzer.scoring_rules,
                take_profit_pct,
                stop_loss_pct
            )
            price_data, entries, exits = strategy.generate_vbt_signals()

            if not price_data.empty and strategy.transaction_log:
                try:
                    # Use the same calculation method as the final analysis
                    final_value = strategy.transaction_log[-1]['portfolio_value']
                    initial_value = strategy.initial_capital
                    total_return = ((final_value - initial_value) / initial_value) * 100
                    
                    # Calculate other metrics from transaction log
                    sell_transactions = [t for t in strategy.transaction_log if t['type'].startswith('SELL')]
                    total_trades = len(strategy.transaction_log)
                    winning_trades = len([t for t in sell_transactions if 'profit_pct' in t and t['profit_pct'] > 0])
                    win_rate = (winning_trades / len(sell_transactions) * 100) if sell_transactions else 0
                    
                    # Calculate max drawdown from portfolio history
                    portfolio_values = [t['portfolio_value'] for t in strategy.transaction_log]
                    max_drawdown = 0
                    peak = portfolio_values[0]
                    for value in portfolio_values:
                        if value > peak:
                            peak = value
                        drawdown = (peak - value) / peak * 100
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                    
                    result = {
                        'weights': weights,
                        'total_return': total_return,
                        'sharpe_ratio': total_return / max_drawdown if max_drawdown > 0 else total_return,  # Simplified Sharpe
                        'max_drawdown': max_drawdown,
                        'win_rate': win_rate,
                        'final_value': final_value
                    }
                    self.results.append(result)

                    # Update best result
                    if (self.best_result is None or
                        (result['total_return'] > self.best_result['total_return']) or
                        (result['total_return'] == self.best_result['total_return'] and
                         result['sharpe_ratio'] > self.best_result['sharpe_ratio'])):
                        self.best_result = result

                except Exception as e:
                    st.warning(f"Error testing {weights}: {str(e)}")

        progress_bar.empty()
        status_text.empty()
        return self.best_result, self.results

# Streamlit App
def main():
    st.markdown("""
        <style>
        /* Target the white dialog box inside the overlay */
        div[data-testid="stDialog"] div[role="dialog"] {
            width: 800px !important;   /* Set custom width */
            max-width: 90% !important; /* Optional: responsive max width */
        }
        </style>
    """, unsafe_allow_html=True)


    if 'modal_shown' not in st.session_state:
        st.session_state.modal_shown = False

    @st.dialog("Quick Start: Trading Strategy Optimizer")
    def show_welcome_modal():
        st.write("The Trading Strategy Optimizer turns complex market data into a single trading score, helping traders act with clarity. Before running an optimization, set your **Buy Score Threshold** to control trade entries, define your **Take Profit %** and **Stop Loss %** to manage gains and risks, and choose the **Max Combinations to Test** to balance speed with thoroughness.")
        
        st.write("Once you start, the system will scan scenarios and show which settings deliver the best performance for your goals.")
        
        if st.button("Get Started", type="primary", use_container_width=True):
            st.session_state.modal_shown = True
            st.rerun()

    if not st.session_state.modal_shown:
        st.session_state.modal_shown = True
        show_welcome_modal()


    # Sidebar for user inputs
    with st.sidebar:
        # Header with icon
        st.markdown("""
        <div style="margin: 0; padding: 0;">
            <h1 style="font-size: 24px; font-weight: 700; color: #1f2937; margin: 0; padding: 0;">
                Trading Strategy Optimizer
            </h1>
            <p style="font-size: 14px; color: #6b7280; margin: 0; padding: 0;">
                Optimize and backtest algorithmic trading strategies with multiple stocks
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Scenario Setup
        st.markdown('<h3 class="sidebar-header">Scenario Setup</h3>', unsafe_allow_html=True)
        
        # Date range inputs
        end_date = st.date_input("End Date", value=datetime.now().date())
        start_date = st.date_input("Start Date", value=(datetime.now() - timedelta(days=365)).date())
        
        # Stock Selection
        st.markdown('<h3 class="sidebar-header">Stock Selection</h3>', unsafe_allow_html=True)
        
        ticker_input = st.text_area(
            "Stock Tickers", 
            value="AAPL, GOOGL, MSFT, AMZN", 
            height=100,
            help="Enter stock symbols separated by commas, semicolons, or spaces.\nExample: AAPL, GOOGL, MSFT or AAPL GOOGL MSFT"
        )
        
        # Parse and display tickers
        tickers = parse_tickers(ticker_input)
        if tickers:
            # Display tickers as styled tags
            ticker_tags = " ".join([f'<span style="display: inline-block; background: #dbeafe; color: #1e40af; padding: 2px 4px; border-radius: 4px; font-size: 12px; font-weight: 500; margin: 0; border: 1px solid #93c5fd;">{ticker}</span>' for ticker in tickers])
            st.markdown(f"""
            <div style="margin: 0; padding: 0;">
                <p style="font-size: 12px; color: #1e40af; font-weight: 500; margin: 0; padding: 0;">✓ Selected tickers:</p>
                <div style="margin: 0; padding: 0;">{ticker_tags}</div>
                <p style="font-size: 12px; color: #6b7280; margin: 0; padding: 0;">Total: {len(tickers)} stocks</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Please enter at least one valid ticker symbol")
        
        # Strategy Parameters
        # Strategy Parameters
        st.markdown('<h3 class="sidebar-header">Strategy Parameters</h3>', unsafe_allow_html=True)

    



        # Sliders
        buy_threshold = st.slider("Buy Score Threshold", min_value=0.0, max_value=2.0, value=0.6, step=0.1)
        take_profit_pct = st.slider("Take Profit %", min_value=1, max_value=50, value=10, step=1)
        stop_loss_pct = st.slider("Stop Loss %", min_value=1, max_value=20, value=5, step=1)

        
        # Optimization Settings
        st.markdown('<h3 class="sidebar-header">Optimization Settings</h3>', unsafe_allow_html=True)
        
        run_optimization = st.checkbox("Run Weight Optimization", value=True)
        max_combinations = st.slider("Max Combinations to Test", min_value=1, max_value=200, value=50)
        
        # Analysis Options
        st.markdown('<h3 class="sidebar-header">Analysis Options</h3>', unsafe_allow_html=True)
        
        show_individual_charts = st.checkbox("Show Individual Stock Charts", value=False)
        show_correlation_matrix = st.checkbox("Show Correlation Matrix", value=True)
        
        # Run analysis button
        run_analysis =st.button("Run Optimization", type="primary", use_container_width=True)
       

    # Main panel
    if run_analysis and tickers:
        with st.spinner("Loading data and running analysis..."):
            # Convert dates to strings
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Initialize analyzer
            analyzer = StockAnalyzer(tickers, start_date_str, end_date_str)
            
            # Run optimization if selected
            if run_optimization:
                
                optimizer = StrategyOptimizer(analyzer)
                best_result, all_results = optimizer.optimize_weights(
                    max_combinations=max_combinations,
                    take_profit_pct=take_profit_pct,
                    stop_loss_pct=stop_loss_pct,
                    buy_threshold=buy_threshold
                )
                
                if best_result:
                    # Store best_result for later use
                    best_result_stored = best_result if run_optimization else None
                    
                    # Set optimal weights
                    best_weights = best_result['weights']
                    analyzer.scoring_rules['predicted_return']['weight'] = best_weights[0]
                    analyzer.scoring_rules['confidence_interval']['weight'] = best_weights[1]
                    analyzer.scoring_rules['rsi']['weight'] = best_weights[2]
                    analyzer.scoring_rules['bb_position']['weight'] = best_weights[3]
                    analyzer.scoring_rules['sma_cross']['weight'] = best_weights[4]
                    
                if run_optimization and 'best_result' in locals() and best_result:
                    st.markdown('<h2 class="section-header">Trading Signal Weighting Distribution</h2>', unsafe_allow_html=True)
                    
                    # Recreate weights_df here
                    best_weights = best_result['weights']
                    weights_df = pd.DataFrame({
                        'Indicator': ['Predicted Return', 'Confidence Interval', 'RSI', 'Bollinger Bands', 'SMA Cross'],
                        'Weight': best_weights,
                        'Percentage': [f"{w*100:.1f}%" for w in best_weights]
                    })
                    
                    # Cards in one row (5 columns)
                    cols = st.columns(5)
                    
                    # Create styled cards for all indicators in one row
                    for i, row in weights_df.iterrows():
                        indicator = row['Indicator']
                        weight = row['Weight']
                        percentage = row['Percentage']
                        
                        with cols[i]:
                            st.markdown(f"""
                            <div style="
                                background: white;
                                border-radius: 12px;
                                padding: 20px;
                                text-align: center;
                                margin: 8px;
                                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                                transition: all 0.3s ease;
                                border: 1px solid #e2e8f0;
                                border-left: 4px solid #10b981;
                                min-height: 120px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;
                            ">
                                <div style="font-size: 14px; font-weight: 600; color: #495057; margin-bottom: 8px;">{indicator}</div>
                                <div style="font-size: 24px; font-weight: 700; color: #374151; margin: 4px 0;">{percentage}</div>
                                <div style="font-size: 12px; color: #6c757d; font-weight: 400;">Weight: {weight:.3f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                   
            # Run final analysis with optimal/default weights
            analyzer.analyze_stocks()
                 # Run strategy backtest
        
            strategy = VectorbtTradingStrategy(
                analyzer.processed_data,
                {'buy': buy_threshold},
                analyzer.scoring_rules,
                take_profit_pct,
                stop_loss_pct
            )
            price_data, entries, exits = strategy.generate_vbt_signals()
            
            # Initialize portfolio_df and other variables
            portfolio_df = None
            fig_portfolio = None
            initial_value = strategy.initial_capital
            final_value = initial_value
            total_return = 0
            
            if not price_data.empty and strategy.transaction_log:
                # Get the final portfolio value from the last transaction (single source of truth)
                final_value = strategy.transaction_log[-1]['portfolio_value']
                total_return = ((final_value - initial_value) / initial_value) * 100
                
                # Create portfolio performance chart using transaction log data for consistency
                transaction_df = pd.DataFrame(strategy.transaction_log)
                transaction_df['date'] = pd.to_datetime(transaction_df['date'])
                
                # Create portfolio performance data from transaction log - ensure it includes the final value
                portfolio_df = transaction_df[['date', 'portfolio_value']].copy()
                
                # Keep only the last transaction per date to avoid duplicates, but ensure final value is included
                portfolio_df = portfolio_df.groupby('date').last().reset_index().sort_values('date')
                
                # Ensure the final value matches exactly
                if not portfolio_df.empty:
                    portfolio_df.loc[portfolio_df.index[-1], 'portfolio_value'] = final_value
                
                fig_portfolio = go.Figure()
                fig_portfolio.add_trace(go.Scatter(
                    x=portfolio_df['date'], 
                    y=portfolio_df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='green', width=2)
                ))
                
                # Core Performance Metrics Card with hover effects
                st.markdown('<h2 class="section-header">Core Performance Metrics</h2>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div style="
                        background: white;
                        border-radius: 12px;
                        padding: 20px;
                        text-align: center;
                        margin: 8px;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                        transition: all 0.3s ease;
                        border: 1px solid #e2e8f0;
                        border-left: 4px solid #3b82f6;
                        min-height: 120px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    ">
                        <div style="font-size: 14px; font-weight: 600; color: #495057; margin-bottom: 8px;">Initial Capital</div>
                        <div style="font-size: 24px; font-weight: 700; color: #374151; margin: 4px 0;">${initial_value:,.0f}</div>
                        <div style="font-size: 12px; color: #6c757d; font-weight: 400;">Starting Value</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="
                        background: white;
                        border-radius: 12px;
                        padding: 20px;
                        text-align: center;
                        margin: 8px;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                        transition: all 0.3s ease;
                        border: 1px solid #e2e8f0;
                        border-left: 4px solid #3b82f6;
                        min-height: 120px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    ">
                        <div style="font-size: 14px; font-weight: 600; color: #495057; margin-bottom: 8px;">Final Value</div>
                        <div style="font-size: 24px; font-weight: 700; color: #374151; margin: 4px 0;">${final_value:,.0f}</div>
                        <div style="font-size: 12px; color: #6c757d; font-weight: 400;">Current Value</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="
                        background: white;
                        border-radius: 12px;
                        padding: 20px;
                        text-align: center;
                        margin: 8px;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                        transition: all 0.3s ease;
                        border: 1px solid #e2e8f0;
                        border-left: 4px solid #3b82f6;
                        min-height: 120px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    ">
                        <div style="font-size: 14px; font-weight: 600; color: #495057; margin-bottom: 8px;">Total Return</div>
                        <div style="font-size: 24px; font-weight: 700; color: #374151; margin: 4px 0;">{total_return:.1f}%</div>
                        <div style="font-size: 12px; color: #6c757d; font-weight: 400;">Portfolio Performance</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div style="
                        background: white;
                        border-radius: 12px;
                        padding: 20px;
                        text-align: center;
                        margin: 8px;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                        transition: all 0.3s ease;
                        border: 1px solid #e2e8f0;
                        border-left: 4px solid #3b82f6;
                        min-height: 120px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    ">
                        <div style="font-size: 14px; font-weight: 600; color: #495057; margin-bottom: 8px;">Number of Trades</div>
                        <div style="font-size: 24px; font-weight: 700; color: #374151; margin: 4px 0;">{strategy.trade_count}</div>
                        <div style="font-size: 12px; color: #6c757d; font-weight: 400;">Total Trades</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Advanced Performance Metrics Card with hover effects
                if run_optimization and best_result:
                    st.markdown('<h2 class="section-header">Advanced Performance Metrics</h2>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="
                            background: white;
                            border-radius: 12px;
                            padding: 20px;
                            text-align: center;
                            margin: 8px;
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                            transition: all 0.3s ease;
                            border: 1px solid #e2e8f0;
                            border-left: 4px solid #10b981;
                            min-height: 120px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                        ">
                            <div style="font-size: 14px; font-weight: 600; color: #495057; margin-bottom: 8px;">Best Total Return</div>
                            <div style="font-size: 24px; font-weight: 700; color: #374151; margin: 4px 0;">{best_result['total_return']:.1f}%</div>
                            <div style="font-size: 12px; color: #6c757d; font-weight: 400;">Optimized Performance</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="
                            background: white;
                            border-radius: 12px;
                            padding: 20px;
                            text-align: center;
                            margin: 8px;
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                            transition: all 0.3s ease;
                            border: 1px solid #e2e8f0;
                            border-left: 4px solid #3b82f6;
                            min-height: 120px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                        ">
                            <div style="font-size: 14px; font-weight: 600; color: #495057; margin-bottom: 8px;">Sharpe Ratio</div>
                            <div style="font-size: 24px; font-weight: 700; color: #374151; margin: 4px 0;">{best_result['sharpe_ratio']:.2f}</div>
                            <div style="font-size: 12px; color: #6c757d; font-weight: 400;">Risk-Adjusted Return</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div style="
                            background: white;
                            border-radius: 12px;
                            padding: 20px;
                            text-align: center;
                            margin: 8px;
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                            transition: all 0.3s ease;
                            border: 1px solid #e2e8f0;
                            border-left: 4px solid #10b981;
                            min-height: 120px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                        ">
                            <div style="font-size: 14px; font-weight: 600; color: #495057; margin-bottom: 8px;">Max Drawdown</div>
                            <div style="font-size: 24px; font-weight: 700; color: #374151; margin: 4px 0;">{best_result['max_drawdown']:.1f}%</div>
                            <div style="font-size: 12px; color: #6c757d; font-weight: 400;">Maximum Loss</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div style="
                            background: white;
                            border-radius: 12px;
                            padding: 20px;
                            text-align: center;
                            margin: 8px;
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                            transition: all 0.3s ease;
                            border: 1px solid #e2e8f0;
                            border-left: 4px solid #3b82f6;
                            min-height: 120px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                        ">
                            <div style="font-size: 14px; font-weight: 600; color: #495057; margin-bottom: 8px;">Win Rate</div>
                            <div style="font-size: 24px; font-weight: 700; color: #374151; margin: 4px 0;">{best_result['win_rate']:.1f}%</div>
                            <div style="font-size: 12px; color: #6c757d; font-weight: 400;">Success Rate</div>
                        </div>
                        """, unsafe_allow_html=True)
                
            
            if not analyzer.processed_data:
                st.error("No valid data was processed for any of the selected tickers.")
                return
            
            # Show weight distribution and correlation matrix in one row (moved above Portfolio Overview)
            
            # Bar chart representation of optimal weights
            if run_optimization and best_result:
                weights_df = pd.DataFrame({
                    'Indicator': ['Predicted Return', 'Confidence Interval', 'RSI', 'Bollinger Bands', 'SMA Cross'],
                    'Weight': best_weights,
                    'Percentage': [f"{w*100:.1f}%" for w in best_weights]
                })
                
                # Create correlation matrix if requested
                if show_correlation_matrix and len(analyzer.tickers) > 1:
                    # Create correlation matrix
                    price_data = {}
                    for ticker in analyzer.tickers:
                        if ticker in analyzer.processed_data:
                            price_data[ticker] = analyzer.processed_data[ticker]['Close']
                    
                    if len(price_data) > 1:
                        corr_df = pd.DataFrame(price_data).corr()
                                 
                        # Weight Distribution Chart in separate row
                        st.markdown('<h2 class="section-header">Weight Distribution Chart</h2>', unsafe_allow_html=True)
                        chart_html = create_chartjs_weight_distribution(weights_df)
                        st.components.v1.html(chart_html, height=400)
                        
                        # Add some CSS to reduce spacing between components
                        st.markdown("""
                        <style>
                        /* Target Streamlit HTML components more specifically */
                        .stComponentsV1Html, 
                        div[data-testid="stHorizontalBlock"] > div,
                        .element-container {
                            margin-bottom: 0 !important;
                            padding-bottom: 0 !important;
                        }
                        
                        /* Reduce spacing after the weight distribution chart */
                        .stComponentsV1Html:first-of-type {
                            margin-bottom: 8px !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Correlation Matrix in separate row with reduced top margin
                        st.markdown('<h2 class="section-header" style="margin-top: -32px;">Stock Price Correlation Matrix</h2>', unsafe_allow_html=True)
                        matrix_html, total_height = create_html_correlation_matrix(corr_df)
                        st.components.v1.html(matrix_html, height=total_height)
            
            # Display portfolio overview
            st.markdown('<h2 class="section-header">Portfolio Overview</h2>', unsafe_allow_html=True)
            
            # Calculate portfolio metrics
            portfolio_metrics = []
            for ticker in analyzer.tickers:
                if ticker in analyzer.processed_data and not analyzer.processed_data[ticker].empty:
                    data = analyzer.processed_data[ticker]
                    last_price = data['Close'].iloc[-1]
                    first_price = data['Close'].iloc[0]
                    buy_hold_return = ((last_price - first_price) / first_price) * 100
                    current_score = data['Total_Score'].iloc[-1] if not data['Total_Score'].empty else 0
                    
                    portfolio_metrics.append({
                        'Ticker': ticker,
                        'Current Price': f"${last_price:,.2f}",
                        'Buy & Hold Return': f"{buy_hold_return:.2f}%",
                        'Current Score': f"{current_score:.2f}",
                        'Data Points': f"{len(data):,}"
                    })
            
            if portfolio_metrics:
                portfolio_overview_df = pd.DataFrame(portfolio_metrics)
                
                # Portfolio Overview Card with hover effects
            
                st.dataframe(portfolio_overview_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
                # Add buy/sell markers for all tickers
                transactions_df = pd.DataFrame(strategy.transaction_log)
                if not transactions_df.empty:
                    buy_transactions = transactions_df[transactions_df['type'] == 'BUY']
                    sell_transactions = transactions_df[transactions_df['type'].str.startswith('SELL')]
                    
                    if not buy_transactions.empty:
                        fig_portfolio.add_trace(go.Scatter(
                            x=buy_transactions['date'],
                            y=buy_transactions['portfolio_value'],
                            mode='markers',
                            name='Buy Orders',
                            marker=dict(color='blue', size=10, symbol='triangle-up'),
                            text=buy_transactions['ticker'],
                            hovertemplate='<b>%{text}</b><br>Buy: $%{y:,.0f}<br>%{x}<extra></extra>'
                        ))
                    
                    if not sell_transactions.empty:
                        fig_portfolio.add_trace(go.Scatter(
                            x=sell_transactions['date'],
                            y=sell_transactions['portfolio_value'],
                            mode='markers',
                            name='Sell Orders',
                            marker=dict(color='red', size=10, symbol='triangle-down'),
                            text=sell_transactions['ticker'],
                            hovertemplate='<b>%{text}</b><br>Sell: $%{y:,.0f}<br>%{x}<extra></extra>'
                        ))
                
                # Portfolio Performance Chart Card with hover effects
                st.markdown('<h2 class="section-header">Portfolio Performance</h2>', unsafe_allow_html=True)
                
                if portfolio_df is not None and not portfolio_df.empty:
                    # Use Chart.js
                    chart_html = create_chartjs_portfolio_performance(portfolio_df)
                    st.components.v1.html(chart_html, height=500)
                else:
                    st.info("No portfolio performance data available. Run the analysis to see portfolio performance.")
                
                
                # Display transaction log
                if strategy.transaction_log:
                    # Create a copy for calculations (keep numeric values)
                    df_transactions_calc = pd.DataFrame(strategy.transaction_log)
                    df_transactions_calc['date'] = df_transactions_calc['date'].dt.strftime('%Y-%m-%d')
                    
                    # Create a copy for display (format as strings)
                    df_transactions_display = df_transactions_calc.copy()
                    df_transactions_display['amount'] = df_transactions_display['amount'].apply(lambda x: f"${x:,.0f}")
                    df_transactions_display['fee'] = df_transactions_display['fee'].apply(lambda x: f"${x:,.0f}")
                    df_transactions_display['cash_balance'] = df_transactions_display['cash_balance'].apply(lambda x: f"${x:,.0f}")
                    df_transactions_display['portfolio_value'] = df_transactions_display['portfolio_value'].apply(lambda x: f"${x:,.0f}")
                    
                    # Format profit percentage if it exists
                    if 'profit_pct' in df_transactions_display.columns:
                        df_transactions_display['profit_pct'] = df_transactions_display['profit_pct'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
                    
                    # Add color coding for transaction types
                    def color_transactions(val):
                        if val == 'BUY':
                            return 'background-color: lightblue'
                        elif val.startswith('SELL'):
                            return 'background-color: lightcoral'
                        return ''
                    
                    styled_df = df_transactions_display.style.applymap(color_transactions, subset=['type'])
                    
                    # Transaction Log Card with hover effects
                    st.markdown('<h2 class="section-header">Transaction Log</h2>', unsafe_allow_html=True)
                    st.dataframe(styled_df, use_container_width=True)
                    
                
                    # Transaction summary by ticker (use numeric values for calculations)
                    ticker_summary = df_transactions_calc.groupby('ticker').agg({
                        'type': 'count',
                        'amount': 'sum',
                        'fee': 'sum'
                    }).rename(columns={'type': 'total_transactions'})
                    
                    # Calculate profit/loss by ticker using actual profit data from transaction log
                    ticker_pl = {}
                    ticker_pl_pct = {}
                    for ticker in df_transactions_calc['ticker'].unique():
                        ticker_trades = df_transactions_calc[df_transactions_calc['ticker'] == ticker]
                        
                        # Sum up actual profits from sell transactions using profit_pct data
                        sell_trades = ticker_trades[ticker_trades['type'].str.startswith('SELL')]
                        total_profit = 0
                        total_buy_amount = 0
                        
                        for _, trade in sell_trades.iterrows():
                            if 'profit_pct' in trade and pd.notna(trade['profit_pct']):
                                # Find the corresponding buy trade for this ticker
                                buy_trades = ticker_trades[ticker_trades['type'] == 'BUY']
                                if not buy_trades.empty:
                                    # Use the buy amount to calculate actual profit
                                    buy_amount = buy_trades.iloc[0]['amount']  # Use first buy for this cycle
                                    profit_amount = (trade['profit_pct'] / 100) * buy_amount
                                    total_profit += profit_amount
                                    total_buy_amount += buy_amount
                        
                        ticker_pl[ticker] = total_profit
                        ticker_pl_pct[ticker] = (total_profit / total_buy_amount * 100) if total_buy_amount > 0 else 0
                    
                    ticker_summary['profit_loss'] = ticker_summary.index.map(ticker_pl)
                    ticker_summary['profit_loss_pct'] = ticker_summary.index.map(ticker_pl_pct)
                    
                    # Format dollar columns for better readability
                    ticker_summary['amount'] = ticker_summary['amount'].apply(lambda x: f"${x:,.0f}")
                    ticker_summary['fee'] = ticker_summary['fee'].apply(lambda x: f"${x:,.0f}")
                    ticker_summary['profit_loss'] = ticker_summary['profit_loss'].apply(lambda x: f"${x:,.0f}")
                    ticker_summary['profit_loss_pct'] = ticker_summary['profit_loss_pct'].apply(lambda x: f"{x:.1f}%")
                    
                    # Add a summary row showing final portfolio value
                    summary_row = pd.DataFrame({
                        'total_transactions': [len(strategy.transaction_log)],
                        'amount': [f"${final_value:,.0f}"],
                        'fee': [f"$"],
                        'profit_loss': [f"${final_value - initial_value:,.0f}"],
                        'profit_loss_pct': [f"{total_return:.1f}%"]
                    }, index=['PORTFOLIO TOTAL'])
                    
                    # Combine ticker summary with portfolio total
                    ticker_summary_with_total = pd.concat([ticker_summary])
                    
                    # Transaction Summary Card with hover effects
                    st.markdown('<h2 class="section-header">Transaction Summary by Ticker</h2>', unsafe_allow_html=True)
                    st.dataframe(ticker_summary_with_total, use_container_width=True)
                else:
                    st.info("No transactions were executed with current parameters")
                
                # Show individual stock charts if requested
                if show_individual_charts:
                    st.markdown('<h2 class="section-header">Individual Stock Analysis</h2>', unsafe_allow_html=True)
                    
                    # Create tabs for each stock
                    if len(analyzer.tickers) > 1:
                        tabs = st.tabs(analyzer.tickers)
                        for i, ticker in enumerate(analyzer.tickers):
                            with tabs[i]:
                                display_individual_stock_chart(analyzer.processed_data[ticker], ticker, buy_threshold)
                    else:
                        # Single stock
                        ticker = analyzer.tickers[0]
                        display_individual_stock_chart(analyzer.processed_data[ticker], ticker, buy_threshold)
                        
            else:
                st.warning("No trades were executed with the current parameters. Try adjusting the buy threshold or other settings.")
    
    elif run_analysis and not tickers:
        st.error("Please enter at least one valid ticker symbol before running the analysis.")

def display_individual_stock_chart(data, ticker, buy_threshold):
    """Display individual stock analysis chart"""
    if data.empty:
        st.error(f"No data available for {ticker}")
        return
    
    # Calculate metrics
    last_price = data['Close'].iloc[-1]
    first_price = data['Close'].iloc[0]
    buy_hold_return = ((last_price - first_price) / first_price) * 100
    current_score = data['Total_Score'].iloc[-1] if not data['Total_Score'].empty else 0
    

    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${last_price:,.2f}")
    with col2:
        st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
    with col3:
        st.metric("Current Score", f"{current_score:.2f}")
    
    # Create technical analysis chart
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'{ticker} Price & Moving Averages', 'RSI', 'Total Score'],
        vertical_spacing=0.08,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price chart
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
    
    if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # RSI chart
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Score chart
    fig.add_trace(go.Scatter(x=data.index, y=data['Total_Score'], name='Total Score', line=dict(color='green')), row=3, col=1)
    fig.add_hline(y=buy_threshold, line_dash="dash", line_color="red", row=3, col=1, annotation_text="Buy Threshold")
    
    fig.update_layout(
        height=800,
        width=1400,  # Further increased width to accommodate legend
        title_text=f"Technical Analysis for {ticker}", 
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.98,
            font=dict(size=10)  # Smaller font size for legend
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=100, b=50, l=50, r=50)  # Add top margin for legend
    )
    
    
    
    st.plotly_chart(fig, use_container_width=True)



def create_chartjs_weight_distribution(weights_df):
    """Create Chart.js weight distribution chart HTML"""
    labels = weights_df['Indicator'].tolist()
    data = weights_df['Weight'].tolist()
    colors = ['#10b981', '#3b82f6', '#10b981', '#3b82f6', '#6366f1']
    
    chart_html = f"""
    <div style="background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
        <div style="height: 300px; position: relative;">
            <canvas id="weightChart"></canvas>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const weightCtx = document.getElementById('weightChart').getContext('2d');
        new Chart(weightCtx, {{
            type: 'bar',
            data: {{
                labels: {labels},
                datasets: [{{
                    label: 'Weight %',
                    data: {data},
                    backgroundColor: {colors},
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1,
                        ticks: {{
                            callback: function(value) {{
                                return (value * 100).toFixed(1) + '%';
                            }}
                        }}
                    }},
                    x: {{
                        grid: {{
                            display: false
                        }}
                    }}
                }}
            }}
        }});
    </script>
    """
    return chart_html

def create_chartjs_portfolio_performance(portfolio_df):
    """Create Chart.js portfolio performance chart HTML"""
    # Check if the required columns exist
    if 'date' not in portfolio_df.columns or 'portfolio_value' not in portfolio_df.columns:
        return """
        <div style="background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
            <h3 style="font-size: 18px; font-weight: 600; color: #1f2937; margin-bottom: 16px; font-family: sans-serif;">Portfolio Performance</h3>
            <div style="height: 400px; display: flex; align-items: center; justify-content: center;">
                <p style="color: #6b7280;">No portfolio data available</p>
            </div>
        </div>
        """
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(portfolio_df['date']):
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    
    dates = portfolio_df['date'].dt.strftime('%Y-%m-%d').tolist()
    values = portfolio_df['portfolio_value'].tolist()
    
    chart_html = f"""
    <div style="background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
        <h3 style="font-size: 18px; font-weight: 600; color: #1f2937; margin-bottom: 16px; font-family: sans-serif;">Portfolio Performance</h3>
        <div style="height: 400px; position: relative;">
            <canvas id="performanceChart"></canvas>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const performanceCtx = document.getElementById('performanceChart').getContext('2d');
        new Chart(performanceCtx, {{
            type: 'line',
            data: {{
                labels: {dates},
                datasets: [{{
                    label: 'Portfolio Value',
                    data: {values},
                    borderColor: '#374151',
                    backgroundColor: 'rgba(55, 65, 81, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    intersect: false,
                    mode: 'index'
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return 'Portfolio Value: $' + context.parsed.y.toLocaleString();
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        ticks: {{
                            callback: function(value) {{
                                return '$' + (value / 1000).toFixed(0) + 'K';
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
    """
    return chart_html

def create_html_correlation_matrix(corr_df):
    """Create HTML correlation matrix heatmap"""
    tickers = corr_df.columns.tolist()
    
    # Create header row
    header_html = '<div class="correlation-cell bg-gray-100 font-bold"></div>'
    for ticker in tickers:
        header_html += f'<div class="correlation-cell bg-gray-100 font-bold text-xs">{ticker}</div>'
    
    # Create data rows
    rows_html = ''
    for i, ticker in enumerate(tickers):
        row_html = f'<div class="correlation-cell bg-gray-100 font-bold text-xs">{ticker}</div>'
        for j, corr_ticker in enumerate(tickers):
            corr_value = corr_df.iloc[i, j]
            
            # Color coding based on correlation value
            if corr_value == 1.0:
                bg_color = '#059669'  # Green for perfect correlation
            elif corr_value >= 0.8:
                bg_color = '#10b981'  # Light green
            elif corr_value >= 0.6:
                bg_color = '#3b82f6'  # Blue
            elif corr_value >= 0.4:
                bg_color = '#6366f1'  # Indigo
            elif corr_value >= 0.2:
                bg_color = '#8b5cf6'  # Purple
            else:
                bg_color = '#a855f7'  # Dark purple
            
            row_html += f'<div class="correlation-cell text-white" style="background-color: {bg_color};">{corr_value:.2f}</div>'
        rows_html += row_html
    
    # Calculate dynamic height based on number of tickers
    cell_height = 60  # Increased from 40 to 60
    header_height = 60  # Increased from 50 to 60
    matrix_height = (len(tickers) + 1) * cell_height  # +1 for header row
    total_height = header_height + matrix_height + 40  # Increased padding from 32 to 40
    
    matrix_html = f"""
    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); height: {total_height}px;">
        <div style="display: flex; justify-content: center;">
            <div style="display: grid; grid-template-columns: repeat({len(tickers) + 1}, minmax(70px, 1fr)); gap: 0;">
                {header_html}
                {rows_html}
            </div>
        </div>
    </div>
    
    <style>
        .correlation-cell {{
            min-width: 70px;
            max-width: 90px;
            height: {cell_height}px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #e5e7eb;
            font-size: 12px;
            font-weight: 600;
            font-family: sans-serif;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
    </style>
    """
    return matrix_html, total_height

if __name__ == "__main__":
    main()


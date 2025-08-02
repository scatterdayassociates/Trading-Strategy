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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

        pred_ret_score = np.where(data_df.get('Predicted_Return', pd.Series(np.nan, index=data_df.index)) > 0,
                                   scoring_rules['predicted_return']['positive'],
                                   scoring_rules['predicted_return']['negative'])
        anomaly_col = data_df.get('Anomaly_Detected', pd.Series(False, index=data_df.index)).fillna(False)
        zero_pred_score_cond = data_df.get('Predicted_Return', pd.Series(np.nan, index=data_df.index)).isna() | anomaly_col
        pred_ret_score = np.where(zero_pred_score_cond, 0, pred_ret_score)
        scores_df['predicted_return'] = pred_ret_score

        conf_low = data_df.get('Predicted_CI_Low')
        conf_high = data_df.get('Predicted_CI_High')
        conf_low_arr = conf_low.values if conf_low is not None else np.full(len(data_df), np.nan)
        conf_high_arr = conf_high.values if conf_high is not None else np.full(len(data_df), np.nan)

        ci_score = np.select(
            [ (conf_low_arr > 0) & (conf_high_arr > 0), (conf_low_arr < 0) & (conf_high_arr < 0) ],
            [ scoring_rules['confidence_interval']['both_positive'],
              scoring_rules['confidence_interval']['both_negative'] ],
            default=scoring_rules['confidence_interval']['mixed']
        )
        zero_ci_score_cond = np.isnan(conf_low_arr) | np.isnan(conf_high_arr) | anomaly_col
        ci_score = np.where(zero_ci_score_cond, 0, ci_score)
        scores_df['confidence_interval'] = ci_score

        rsi = data_df.get('RSI')
        rsi_arr = rsi.values if rsi is not None else np.full(len(data_df), np.nan)
        rsi_low, rsi_high = scoring_rules['rsi']['range']
        rsi_score = np.select(
            [ rsi_arr < rsi_low, rsi_arr > rsi_high ],
            [ scoring_rules['rsi']['oversold'], scoring_rules['rsi']['overbought'] ],
            default=scoring_rules['rsi']['within_range']
        )
        rsi_score = np.where(np.isnan(rsi_arr), 0, rsi_score)
        scores_df['rsi'] = rsi_score

        bb_pos = data_df.get('BB_Position')
        bb_pos_arr = bb_pos.values if bb_pos is not None else np.full(len(data_df), np.nan)
        bb_low, bb_high = scoring_rules['bb_position']['range']
        bb_pos_score = np.select(
            [ bb_pos_arr < bb_low, bb_pos_arr > bb_high ],
            [ scoring_rules['bb_position']['below_range'], scoring_rules['bb_position']['above_range'] ],
            default=scoring_rules['bb_position']['within_range']
        )
        bb_pos_score = np.where(np.isnan(bb_pos_arr), 0, bb_pos_score)
        scores_df['bb_position'] = bb_pos_score

        sma20 = data_df.get('SMA_20')
        sma50 = data_df.get('SMA_50')
        sma20_arr = sma20.values if sma20 is not None else np.full(len(data_df), np.nan)
        sma50_arr = sma50.values if sma50 is not None else np.full(len(data_df), np.nan)

        sma_cross_score = np.select(
            [ sma20_arr > sma50_arr, sma20_arr < sma50_arr ],
            [ scoring_rules['sma_cross']['sma20_above_sma50'], scoring_rules['sma_cross']['sma20_below_sma50'] ],
            default=0
        )
        sma_cross_score = np.where(np.isnan(sma20_arr) | np.isnan(sma50_arr), 0, sma_cross_score)
        scores_df['sma_cross'] = sma_cross_score

        total_score = pd.Series(0.0, index=data_df.index)
        for indicator_name, rule in scoring_rules.items():
            daily_scores = scores_df[indicator_name].fillna(0)
            total_score += daily_scores * rule['weight']

        data_df_with_scores = data_df.copy()
        data_df_with_scores['Total_Score'] = total_score

        return data_df_with_scores, scores_df

    def analyze_stocks(self):
        self.processed_data = {}

        for ticker in self.tickers:
            data_with_ohlc = self.get_clean_financial_data(ticker)

            if data_with_ohlc is None or data_with_ohlc.empty or \
               'Close' not in data_with_ohlc.columns or data_with_ohlc['Close'].empty or \
               not isinstance(data_with_ohlc['Close'], pd.Series):
                st.error(f"Critical failure: Could not process valid 'Close' Series for {ticker}.")
                self.processed_data[ticker] = pd.DataFrame()
                continue

            data_with_ohlc['Daily_Return'] = data_with_ohlc['Close'].pct_change()
            data_with_ti = self.calculate_technical_indicators(data_with_ohlc)
            data_with_pred = self.calculate_rolling_prediction_and_anomaly(data_with_ti)

            if data_with_pred is None or data_with_pred.empty:
                st.error(f"Processing resulted in empty data after prediction calc for {ticker}.")
                self.processed_data[ticker] = pd.DataFrame()
                continue

            data_final, _ = self.calculate_daily_scores(data_with_pred, self.scoring_rules)

            if data_final is None or data_final.empty:
                st.error(f"Processing resulted in empty data after score calc for {ticker}.")
                self.processed_data[ticker] = pd.DataFrame()
                continue

            self.processed_data[ticker] = data_final

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
        self.shares_held = 0
        self.position_value = 0
        self.fee_rate = 0.001  # 0.1% transaction fee
        self.trade_count = 0
        self.first_trade_date = None
        self.last_trade_date = None

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

            in_position = False
            entry_price = 0
            entry_date = None

            for i in range(len(data_df)):
                current_date = data_df.index[i]
                current_price = close_prices.iloc[i]

                # Calculate current portfolio value
                current_portfolio_value = self.shares_held * current_price + self.cash_balance
                self.portfolio_history.append({
                    'date': current_date,
                    'portfolio_value': current_portfolio_value,
                    'cash': self.cash_balance,
                    'position_value': self.shares_held * current_price,
                    'shares': self.shares_held
                })

                # Entry logic
                if not in_position:
                    score = data_df['Total_Score'].iloc[i]
                    if score >= self.decision_thresholds['buy']:
                        # Calculate position size (100% of portfolio)
                        position_value = self.cash_balance
                        shares = int(position_value / current_price)
                        if shares == 0:  # Not enough cash
                            continue

                        cost = shares * current_price
                        fee = cost * self.fee_rate
                        total_cost = cost + fee

                        if total_cost > self.cash_balance:
                            continue

                        # Execute buy
                        entries.iloc[i] = True
                        in_position = True
                        entry_price = current_price
                        entry_date = current_date
                        self.shares_held = shares
                        self.cash_balance -= total_cost
                        self.trade_count += 1

                        # Update first trade date if needed
                        if self.first_trade_date is None or current_date < self.first_trade_date:
                            self.first_trade_date = current_date

                        # Log transaction
                        self.transaction_log.append({
                            'date': current_date,
                            'type': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'amount': cost,
                            'fee': fee,
                            'cash_balance': self.cash_balance,
                            'portfolio_value': current_portfolio_value,
                            'position_size_pct': 100
                        })

                # Exit logic
                if in_position:
                    returns = (current_price - entry_price) / entry_price

                    # Take profit
                    if returns >= self.take_profit_pct / 100:
                        exits.iloc[i] = True
                        in_position = False
                        proceeds = self.shares_held * current_price
                        fee = proceeds * self.fee_rate
                        net_proceeds = proceeds - fee
                        self.cash_balance += net_proceeds
                        self.last_trade_date = current_date

                        # Log transaction
                        self.transaction_log.append({
                            'date': current_date,
                            'type': 'SELL-TP',
                            'price': current_price,
                            'shares': self.shares_held,
                            'amount': proceeds,
                            'fee': fee,
                            'cash_balance': self.cash_balance,
                            'portfolio_value': self.cash_balance,
                            'profit_pct': returns * 100
                        })
                        self.shares_held = 0
                        self.trade_count += 1

                    # Stop loss
                    elif returns <= -self.stop_loss_pct / 100:
                        exits.iloc[i] = True
                        in_position = False
                        proceeds = self.shares_held * current_price
                        fee = proceeds * self.fee_rate
                        net_proceeds = proceeds - fee
                        self.cash_balance += net_proceeds
                        self.last_trade_date = current_date

                        # Log transaction
                        self.transaction_log.append({
                            'date': current_date,
                            'type': 'SELL-SL',
                            'price': current_price,
                            'shares': self.shares_held,
                            'amount': proceeds,
                            'fee': fee,
                            'cash_balance': self.cash_balance,
                            'portfolio_value': self.cash_balance,
                            'profit_pct': returns * 100
                        })
                        self.shares_held = 0
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

    def optimize_weights(self, max_combinations=50, take_profit_pct=10, stop_loss_pct=5):
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

            # Re-analyze and backtest
            self.analyzer.analyze_stocks()
            strategy = VectorbtTradingStrategy(
                self.analyzer.processed_data,
                {'buy': 0.6},
                self.analyzer.scoring_rules,
                take_profit_pct,
                stop_loss_pct
            )
            price_data, entries, exits = strategy.generate_vbt_signals()

            if not price_data.empty:
                try:
                    portfolio = vbt.Portfolio.from_signals(
                        close=price_data,
                        entries=entries,
                        exits=exits,
                        size=1,
                        size_type='amount',
                        init_cash=100000,
                        fees=0.001,
                        direction='longonly',
                        freq='d'
                    )

                    stats = portfolio.stats()
                    result = {
                        'weights': weights,
                        'total_return': stats.loc['Total Return [%]'],
                        'sharpe_ratio': stats.loc['Sharpe Ratio'],
                        'max_drawdown': stats.loc['Max Drawdown [%]'],
                        'win_rate': stats.loc['Win Rate [%]'],
                        'stats': stats
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
    st.title("Trading Strategy Optimizer")
    st.markdown("Optimize and backtest algorithmic trading strategies with technical indicators")

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Configuration")
        
        # Date range inputs (moved from main code lines 2,3,4)
        end_date = st.date_input("End Date", value=datetime.now().date())
        start_date = st.date_input("Start Date", value=(datetime.now() - timedelta(days=365)).date())
        
        # Ticker input
        ticker_input = st.text_input("Stock Ticker", value="AMZN", help="Enter stock symbol (e.g., AAPL, GOOGL)")
        tickers = [ticker_input.upper().strip()] if ticker_input else ['AMZN']
        
        st.divider()
        
        # Strategy parameters
        st.subheader("Strategy Parameters")
        buy_threshold = st.slider("Buy Score Threshold", min_value=0.0, max_value=2.0, value=0.6, step=0.1)
        take_profit_pct = st.slider("Take Profit %", min_value=1, max_value=50, value=10, step=1)
        stop_loss_pct = st.slider("Stop Loss %", min_value=1, max_value=20, value=5, step=1)
        
        st.divider()
        
        # Optimization settings
        st.subheader("Optimization Settings")
        run_optimization = st.checkbox("Run Weight Optimization", value=True)
        max_combinations = st.slider("Max Combinations to Test", min_value=10, max_value=200, value=50)
        
        st.divider()
        
        # Run analysis button
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

    # Main panel
    if run_analysis:
        with st.spinner("Loading data and running analysis..."):
            # Convert dates to strings
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Initialize analyzer
            analyzer = StockAnalyzer(tickers, start_date_str, end_date_str)
            
            # Run optimization if selected
            if run_optimization:
                st.header("Weight Optimization Results")
                optimizer = StrategyOptimizer(analyzer)
                best_result, all_results = optimizer.optimize_weights(
                    max_combinations=max_combinations,
                    take_profit_pct=take_profit_pct,
                    stop_loss_pct=stop_loss_pct
                )
                
                if best_result:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Best Total Return", f"{best_result['total_return']:.1f}%")
                    with col2:
                        st.metric("Sharpe Ratio", f"{best_result['sharpe_ratio']:.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{best_result['max_drawdown']:.1f}%")
                    with col4:
                        st.metric("Win Rate", f"{best_result['win_rate']:.1f}%")
                    
                    # Set optimal weights
                    best_weights = best_result['weights']
                    analyzer.scoring_rules['predicted_return']['weight'] = best_weights[0]
                    analyzer.scoring_rules['confidence_interval']['weight'] = best_weights[1]
                    analyzer.scoring_rules['rsi']['weight'] = best_weights[2]
                    analyzer.scoring_rules['bb_position']['weight'] = best_weights[3]
                    analyzer.scoring_rules['sma_cross']['weight'] = best_weights[4]
                    
                    # Display optimal weights
                    st.subheader("Optimal Weights")
                    weights_df = pd.DataFrame({
                        'Indicator': ['Predicted Return', 'Confidence Interval', 'RSI', 'Bollinger Bands', 'SMA Cross'],
                        'Weight': best_weights
                    })
                    st.dataframe(weights_df, use_container_width=True)
            
            # Run final analysis with optimal/default weights
            analyzer.analyze_stocks()
            
            # Display stock data and metrics
            st.header(f"Analysis Results for {tickers[0]}")
            
            if tickers[0] in analyzer.processed_data and not analyzer.processed_data[tickers[0]].empty:
                data = analyzer.processed_data[tickers[0]]
                
                # Calculate buy and hold return
                last_price = data['Close'].iloc[-1]
                first_price = data['Close'].iloc[0]
                buy_hold_return = ((last_price - first_price) / first_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${last_price:.2f}")
                with col2:
                    st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
                with col3:
                    st.metric("Data Points", len(data))
                
                # Create price chart with technical indicators
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=['Price & Moving Averages', 'RSI', 'Total Score'],
                    vertical_spacing=0.08,
                    row_heights=[0.6, 0.2, 0.2]
                )
                
                # Price chart
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
                
                # RSI chart
                fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # Score chart
                fig.add_trace(go.Scatter(x=data.index, y=data['Total_Score'], name='Total Score', line=dict(color='green')), row=3, col=1)
                fig.add_hline(y=buy_threshold, line_dash="dash", line_color="red", row=3, col=1)
                
                fig.update_layout(height=800, title_text=f"Technical Analysis for {tickers[0]}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Run strategy backtest
                st.header("Strategy Backtest Results")
                strategy = VectorbtTradingStrategy(
                    analyzer.processed_data,
                    {'buy': buy_threshold},
                    analyzer.scoring_rules,
                    take_profit_pct,
                    stop_loss_pct
                )
                price_data, entries, exits = strategy.generate_vbt_signals()
                
                if not price_data.empty and strategy.transaction_log:
                    # Create portfolio performance chart
                    portfolio_df = pd.DataFrame(strategy.portfolio_history)
                    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
                    
                    fig_portfolio = go.Figure()
                    fig_portfolio.add_trace(go.Scatter(
                        x=portfolio_df['date'], 
                        y=portfolio_df['portfolio_value'],
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='green', width=2)
                    ))
                    fig_portfolio.update_layout(
                        title="Portfolio Performance Over Time",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=400
                    )
                    st.plotly_chart(fig_portfolio, use_container_width=True)
                    
                    # Calculate final metrics
                    initial_value = 100000
                    final_value = strategy.cash_balance
                    total_return = ((final_value - initial_value) / initial_value) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Initial Capital", f"${initial_value:,.2f}")
                    with col2:
                        st.metric("Final Value", f"${final_value:,.2f}")
                    with col3:
                        st.metric("Total Return", f"{total_return:.1f}%")
                    with col4:
                        st.metric("Number of Trades", strategy.trade_count)
                    
                    # Display transaction log
                    st.subheader("Transaction Log")
                    if strategy.transaction_log:
                        df_transactions = pd.DataFrame(strategy.transaction_log)
                        df_transactions['date'] = df_transactions['date'].dt.strftime('%Y-%m-%d')
                        st.dataframe(df_transactions, use_container_width=True)
                    else:
                        st.info("No transactions were executed with current parameters")
                        
                else:
                    st.warning("No trades were executed with the current parameters. Try adjusting the buy threshold or other settings.")
            else:
                st.error(f"Failed to process data for {tickers[0]}")

if __name__ == "__main__":
    main()
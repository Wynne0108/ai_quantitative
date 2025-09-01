"""
Pairs Trading Strategy Module

This module implements comprehensive pairs trading strategies based on cointegration
analysis and statistical arbitrage principles. It provides tools for identifying
trading opportunities, managing positions, and evaluating performance.

Key Features:
- Cointegration-based pairs identification
- Dynamic spread calculation and normalization
- Multiple entry/exit signal generation methods
- Position sizing and risk management
- Performance tracking and analysis
- Real-time trading signal monitoring

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt

from .cointegration import CointegrationAnalyzer
from .statistical_tests import StatisticalTests


class PairsTradingStrategy:
    """
    Comprehensive pairs trading strategy implementation
    
    This class provides a complete framework for pairs trading including:
    1. Pair selection based on cointegration analysis
    2. Spread calculation and normalization
    3. Trading signal generation with multiple methods
    4. Position management and risk control
    5. Performance evaluation and reporting
    """
    
    def __init__(self,
                 price_data: pd.DataFrame,
                 pair_assets: Tuple[str, str],
                 lookback_period: int = 252,
                 confidence_level: float = 0.05):
        """
        Initialize PairsTradingStrategy
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Time series price data with datetime index
        pair_assets : Tuple[str, str]
            Names of the two assets to trade (asset1, asset2)
        lookback_period : int, default=252
            Lookback period for parameter estimation (trading days)
        confidence_level : float, default=0.05
            Confidence level for statistical tests
        """
        self.price_data = price_data.copy()
        self.asset1, self.asset2 = pair_assets
        self.lookback_period = lookback_period
        self.confidence_level = confidence_level
        
        # Validate inputs
        if self.asset1 not in price_data.columns or self.asset2 not in price_data.columns:
            raise ValueError(f"Assets {self.asset1} and/or {self.asset2} not found in price data")
        
        # Initialize components
        self.cointegration_analyzer = CointegrationAnalyzer(
            price_data[[self.asset1, self.asset2]],
            [self.asset1, self.asset2],
            lookback_period=lookback_period,
            confidence_level=confidence_level
        )
        
        self.statistical_tests = StatisticalTests()
        
        # Strategy parameters (will be set during setup)
        self.hedge_ratio = None
        self.spread_series = None
        self.cointegration_results = None
        
        # Trading parameters
        self.trading_params = {
            'entry_threshold': 2.0,      # Z-score threshold for entry
            'exit_threshold': 0.5,       # Z-score threshold for exit
            'stop_loss_threshold': 4.0,  # Z-score threshold for stop loss
            'lookback_zscore': 60,       # Lookback for z-score calculation
            'min_half_life': 5,          # Minimum mean reversion half-life (days)
            'max_half_life': 60,         # Maximum mean reversion half-life (days)
            'transaction_cost': 0.001,   # Transaction cost per trade (0.1%)
            'position_size': 1.0         # Position size multiplier
        }
        
        # Results storage
        self.signals_data = None
        self.positions_data = None
        self.performance_metrics = None
        
        print(f"PairsTradingStrategy initialized for {self.asset1} vs {self.asset2}")
        print(f"Data period: {price_data.index[0]} to {price_data.index[-1]}")
        print(f"Total observations: {len(price_data)}")
    
    def analyze_pair_relationship(self) -> Dict:
        """
        Analyze the cointegration relationship between the pair
        
        Returns:
        --------
        Dict containing comprehensive pair analysis results
        """
        print("Analyzing pair cointegration relationship...")
        
        # Run cointegration analysis
        coint_results = self.cointegration_analyzer.analyze_pairs_cointegration()
        pair_key = f"{self.asset1}_{self.asset2}"
        
        if pair_key not in coint_results:
            raise ValueError(f"Cointegration analysis failed for pair {pair_key}")
        
        self.cointegration_results = coint_results[pair_key]
        
        # Extract hedge ratio and spread
        if self.cointegration_results['is_cointegrated']:
            best_direction = self.cointegration_results['best_direction']
            best_test = self.cointegration_results[f'test_{best_direction}']
            
            if best_direction == '12':
                # asset1 ~ asset2, so hedge_ratio = beta (asset2 coefficient)
                self.hedge_ratio = best_test['cointegration_coefficient']
                self.spread_series = best_test['spread_series']
            else:
                # asset2 ~ asset1, so hedge_ratio = 1/beta
                self.hedge_ratio = 1.0 / best_test['cointegration_coefficient']
                # Recalculate spread with correct direction
                log_prices = self.cointegration_analyzer.log_prices
                self.spread_series = log_prices[self.asset1] - self.hedge_ratio * log_prices[self.asset2]
        
        # Additional spread analysis
        if self.spread_series is not None:
            spread_analysis = self._analyze_spread_properties()
            self.cointegration_results['spread_analysis'] = spread_analysis
        
        return self.cointegration_results
    
    def _analyze_spread_properties(self) -> Dict:
        """
        Analyze statistical properties of the spread
        
        Returns:
        --------
        Dict containing spread analysis results
        """
        if self.spread_series is None:
            return {'error': 'No spread series available'}
        
        analysis = {}
        
        # Basic statistics
        analysis['descriptive'] = {
            'mean': self.spread_series.mean(),
            'std': self.spread_series.std(),
            'min': self.spread_series.min(),
            'max': self.spread_series.max(),
            'skewness': stats.skew(self.spread_series),
            'kurtosis': stats.kurtosis(self.spread_series)
        }
        
        # Half-life of mean reversion
        try:
            half_life = self._calculate_half_life(self.spread_series)
            analysis['half_life'] = {
                'days': half_life,
                'is_suitable': (self.trading_params['min_half_life'] <= 
                               half_life <= self.trading_params['max_half_life'])
            }
        except Exception as e:
            analysis['half_life'] = {'error': str(e)}
        
        # Stationarity tests
        stationarity = self.statistical_tests.test_stationarity_comprehensive(self.spread_series)
        analysis['stationarity'] = stationarity
        
        # Normality of spread changes
        spread_changes = self.spread_series.diff().dropna()
        normality = self.statistical_tests.test_normality_comprehensive(spread_changes)
        analysis['normality'] = normality
        
        # Autocorrelation
        autocorr = self.statistical_tests.test_autocorrelation(self.spread_series)
        analysis['autocorrelation'] = autocorr
        
        return analysis
    
    def _calculate_half_life(self, series: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process
        
        Parameters:
        -----------
        series : pd.Series
            Time series to analyze
        
        Returns:
        --------
        float: Half-life in days
        """
        # Estimate AR(1) model: delta_y = alpha + beta * y_lag + error
        y_lag = series.shift(1)
        delta_y = series.diff()
        
        # Remove NaN values
        valid_data = pd.concat([delta_y, y_lag], axis=1).dropna()
        y_values = valid_data.iloc[:, 1]  # y_lag
        delta_values = valid_data.iloc[:, 0]  # delta_y
        
        # OLS regression
        X = np.column_stack([np.ones(len(y_values)), y_values])
        coefficients = np.linalg.lstsq(X, delta_values, rcond=None)[0]
        
        beta = coefficients[1]  # Mean reversion speed
        
        # Calculate half-life
        if beta < 0:
            half_life = -np.log(2) / beta
        else:
            # No mean reversion
            half_life = np.inf
        
        return half_life
    
    def set_trading_parameters(self, **params) -> None:
        """
        Update trading parameters
        
        Parameters:
        -----------
        **params : keyword arguments
            Trading parameters to update
        """
        for key, value in params.items():
            if key in self.trading_params:
                self.trading_params[key] = value
                print(f"Updated {key}: {value}")
            else:
                print(f"Warning: Unknown parameter {key}")
    
    def generate_trading_signals(self, 
                                method: str = 'zscore',
                                adaptive_thresholds: bool = False) -> pd.DataFrame:
        """
        Generate trading signals based on spread analysis
        
        Parameters:
        -----------
        method : str, default='zscore'
            Signal generation method ('zscore', 'bollinger', 'percentile')
        adaptive_thresholds : bool, default=False
            Whether to use adaptive thresholds based on volatility regime
        
        Returns:
        --------
        pd.DataFrame containing trading signals and position data
        """
        if self.spread_series is None:
            raise ValueError("Must run analyze_pair_relationship() first")
        
        print(f"Generating trading signals using {method} method...")
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=self.spread_series.index)
        signals['spread'] = self.spread_series
        
        if method == 'zscore':
            signals = self._generate_zscore_signals(signals, adaptive_thresholds)
        elif method == 'bollinger':
            signals = self._generate_bollinger_signals(signals, adaptive_thresholds)
        elif method == 'percentile':
            signals = self._generate_percentile_signals(signals, adaptive_thresholds)
        else:
            raise ValueError(f"Unknown signal generation method: {method}")
        
        # Calculate positions and returns
        signals = self._calculate_positions(signals)
        signals = self._calculate_returns(signals)
        
        self.signals_data = signals
        return signals
    
    def _generate_zscore_signals(self, 
                                signals: pd.DataFrame, 
                                adaptive: bool = False) -> pd.DataFrame:
        """Generate signals based on z-score normalization"""
        lookback = self.trading_params['lookback_zscore']
        
        # Calculate rolling z-score
        rolling_mean = signals['spread'].rolling(window=lookback).mean()
        rolling_std = signals['spread'].rolling(window=lookback).std()
        signals['zscore'] = (signals['spread'] - rolling_mean) / rolling_std
        
        # Set thresholds
        if adaptive:
            # Adaptive thresholds based on volatility regime
            vol_regime = self._identify_volatility_regime(signals['spread'])
            entry_threshold = np.where(vol_regime == 'high', 
                                     self.trading_params['entry_threshold'] * 1.5,
                                     self.trading_params['entry_threshold'])
            exit_threshold = np.where(vol_regime == 'high',
                                    self.trading_params['exit_threshold'] * 1.5,
                                    self.trading_params['exit_threshold'])
        else:
            entry_threshold = self.trading_params['entry_threshold']
            exit_threshold = self.trading_params['exit_threshold']
        
        # Generate signals
        signals['entry_long'] = signals['zscore'] < -entry_threshold
        signals['entry_short'] = signals['zscore'] > entry_threshold
        signals['exit_signal'] = np.abs(signals['zscore']) < exit_threshold
        signals['stop_loss'] = np.abs(signals['zscore']) > self.trading_params['stop_loss_threshold']
        
        return signals
    
    def _generate_bollinger_signals(self, 
                                   signals: pd.DataFrame, 
                                   adaptive: bool = False) -> pd.DataFrame:
        """Generate signals based on Bollinger Bands"""
        lookback = self.trading_params['lookback_zscore']
        
        # Calculate Bollinger Bands
        rolling_mean = signals['spread'].rolling(window=lookback).mean()
        rolling_std = signals['spread'].rolling(window=lookback).std()
        
        multiplier = self.trading_params['entry_threshold']
        signals['bb_upper'] = rolling_mean + multiplier * rolling_std
        signals['bb_lower'] = rolling_mean - multiplier * rolling_std
        signals['bb_middle'] = rolling_mean
        
        # Generate signals
        signals['entry_long'] = signals['spread'] < signals['bb_lower']
        signals['entry_short'] = signals['spread'] > signals['bb_upper']
        signals['exit_signal'] = ((signals['spread'] > signals['bb_middle'] - 0.5 * rolling_std) & 
                                 (signals['spread'] < signals['bb_middle'] + 0.5 * rolling_std))
        
        # Calculate z-score for consistency
        signals['zscore'] = (signals['spread'] - rolling_mean) / rolling_std
        signals['stop_loss'] = np.abs(signals['zscore']) > self.trading_params['stop_loss_threshold']
        
        return signals
    
    def _generate_percentile_signals(self, 
                                    signals: pd.DataFrame, 
                                    adaptive: bool = False) -> pd.DataFrame:
        """Generate signals based on percentile thresholds"""
        lookback = self.trading_params['lookback_zscore']
        
        # Calculate rolling percentiles
        entry_pct = (1 - stats.norm.cdf(self.trading_params['entry_threshold'])) * 100
        exit_pct = (1 - stats.norm.cdf(self.trading_params['exit_threshold'])) * 100
        
        signals['pct_upper'] = signals['spread'].rolling(window=lookback).quantile(1 - entry_pct/100)
        signals['pct_lower'] = signals['spread'].rolling(window=lookback).quantile(entry_pct/100)
        signals['pct_middle'] = signals['spread'].rolling(window=lookback).median()
        
        # Generate signals
        signals['entry_long'] = signals['spread'] < signals['pct_lower']
        signals['entry_short'] = signals['spread'] > signals['pct_upper']
        signals['exit_signal'] = ((signals['spread'] > signals['pct_middle'] - 
                                  (signals['pct_middle'] - signals['pct_lower']) * exit_pct/entry_pct) & 
                                 (signals['spread'] < signals['pct_middle'] + 
                                  (signals['pct_upper'] - signals['pct_middle']) * exit_pct/entry_pct))
        
        # Calculate z-score for consistency
        rolling_mean = signals['spread'].rolling(window=lookback).mean()
        rolling_std = signals['spread'].rolling(window=lookback).std()
        signals['zscore'] = (signals['spread'] - rolling_mean) / rolling_std
        signals['stop_loss'] = np.abs(signals['zscore']) > self.trading_params['stop_loss_threshold']
        
        return signals
    
    def _identify_volatility_regime(self, spread: pd.Series) -> np.ndarray:
        """Identify volatility regime (high/low) for adaptive thresholds"""
        volatility = spread.rolling(window=20).std()
        vol_median = volatility.rolling(window=60).median()
        
        regime = np.where(volatility > vol_median, 'high', 'low')
        return regime
    
    def _calculate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate position signals based on entry/exit rules"""
        signals['position'] = 0
        signals['trade_entry'] = 0
        signals['trade_exit'] = 0
        
        current_position = 0
        
        for i in range(len(signals)):
            # Check for stop loss first
            if current_position != 0 and signals['stop_loss'].iloc[i]:
                signals.iloc[i, signals.columns.get_loc('trade_exit')] = -current_position
                current_position = 0
            
            # Check for regular exit
            elif current_position != 0 and signals['exit_signal'].iloc[i]:
                signals.iloc[i, signals.columns.get_loc('trade_exit')] = -current_position
                current_position = 0
            
            # Check for new entry (only if not currently in position)
            elif current_position == 0:
                if signals['entry_long'].iloc[i]:
                    current_position = 1  # Long spread (long asset1, short asset2)
                    signals.iloc[i, signals.columns.get_loc('trade_entry')] = 1
                elif signals['entry_short'].iloc[i]:
                    current_position = -1  # Short spread (short asset1, long asset2)
                    signals.iloc[i, signals.columns.get_loc('trade_entry')] = -1
            
            signals.iloc[i, signals.columns.get_loc('position')] = current_position
        
        return signals
    
    def _calculate_returns(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy returns and individual asset positions"""
        # Get price data
        prices = self.price_data[[self.asset1, self.asset2]].copy()
        
        # Align signals and prices
        aligned_data = pd.concat([prices, signals], axis=1).dropna()
        
        # Calculate returns
        asset1_returns = aligned_data[self.asset1].pct_change()
        asset2_returns = aligned_data[self.asset2].pct_change()
        
        # Position in each asset (considering hedge ratio)
        spread_position = aligned_data['position'].shift(1).fillna(0)
        
        # For long spread: long asset1, short (hedge_ratio * asset2)
        # For short spread: short asset1, long (hedge_ratio * asset2)
        aligned_data['asset1_position'] = spread_position * self.trading_params['position_size']
        aligned_data['asset2_position'] = -spread_position * self.hedge_ratio * self.trading_params['position_size']
        
        # Calculate individual asset returns
        aligned_data['asset1_return'] = aligned_data['asset1_position'] * asset1_returns
        aligned_data['asset2_return'] = aligned_data['asset2_position'] * asset2_returns
        
        # Total strategy return
        aligned_data['strategy_return'] = aligned_data['asset1_return'] + aligned_data['asset2_return']
        
        # Apply transaction costs
        trades = (aligned_data['position'] != aligned_data['position'].shift(1)).fillna(False)
        transaction_costs = trades * self.trading_params['transaction_cost']
        aligned_data['net_return'] = aligned_data['strategy_return'] - transaction_costs
        
        # Cumulative returns
        aligned_data['cumulative_return'] = (1 + aligned_data['net_return']).cumprod() - 1
        
        # Update signals data
        for col in ['asset1_position', 'asset2_position', 'asset1_return', 'asset2_return', 
                   'strategy_return', 'net_return', 'cumulative_return']:
            if col in aligned_data.columns:
                signals[col] = aligned_data[col]
        
        return signals
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics for the strategy
        
        Returns:
        --------
        Dict containing detailed performance analysis
        """
        if self.signals_data is None:
            raise ValueError("Must generate trading signals first")
        
        returns = self.signals_data['net_return'].dropna()
        
        if len(returns) == 0:
            return {'error': 'No return data available'}
        
        # Basic performance metrics
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1
        
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
        
        # Trade analysis
        trades = self.signals_data['trade_entry'] != 0
        n_trades = trades.sum()
        
        if n_trades > 0:
            # Calculate win rate
            trade_returns = []
            current_entry_date = None
            current_position = 0
            
            for i, row in self.signals_data.iterrows():
                if row['trade_entry'] != 0:
                    current_entry_date = i
                    current_position = row['trade_entry']
                elif row['trade_exit'] != 0 and current_entry_date is not None:
                    # Calculate trade return
                    trade_ret = self.signals_data.loc[current_entry_date:i, 'net_return'].sum()
                    trade_returns.append(trade_ret)
                    current_entry_date = None
                    current_position = 0
            
            if trade_returns:
                win_rate = np.mean([ret > 0 for ret in trade_returns])
                avg_win = np.mean([ret for ret in trade_returns if ret > 0])
                avg_loss = np.mean([ret for ret in trade_returns if ret <= 0])
                profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Information metrics
        benchmark_return = 0  # Assuming risk-free benchmark
        excess_returns = returns - benchmark_return/252
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        performance = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration_days': max_drawdown_duration,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'number_of_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_days': n_periods,
            'start_date': returns.index[0],
            'end_date': returns.index[-1]
        }
        
        self.performance_metrics = performance
        return performance
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for in_drawdown in is_drawdown:
            if in_drawdown:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        # Don't forget the last period if it ends in drawdown
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def plot_strategy_analysis(self, 
                              figsize: Tuple[int, int] = (16, 12),
                              save_path: Optional[str] = None) -> None:
        """
        Create comprehensive strategy analysis plots
        
        Parameters:
        -----------
        figsize : Tuple[int, int], default=(16, 12)
            Figure size for plots
        save_path : str, optional
            Path to save the plot
        """
        if self.signals_data is None:
            raise ValueError("Must generate trading signals first")
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(f'Pairs Trading Strategy: {self.asset1} vs {self.asset2}', fontsize=16)
        
        # Plot 1: Asset prices
        axes[0, 0].plot(self.price_data[self.asset1], label=self.asset1, alpha=0.8)
        axes[0, 0].plot(self.price_data[self.asset2], label=self.asset2, alpha=0.8)
        axes[0, 0].set_title('Asset Prices')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Spread and signals
        axes[0, 1].plot(self.signals_data['spread'], label='Spread', alpha=0.7)
        
        # Mark entry and exit points
        entry_long = self.signals_data[self.signals_data['trade_entry'] == 1]
        entry_short = self.signals_data[self.signals_data['trade_entry'] == -1]
        exits = self.signals_data[self.signals_data['trade_exit'] != 0]
        
        axes[0, 1].scatter(entry_long.index, entry_long['spread'], 
                          color='green', marker='^', s=50, label='Long Entry')
        axes[0, 1].scatter(entry_short.index, entry_short['spread'], 
                          color='red', marker='v', s=50, label='Short Entry')
        axes[0, 1].scatter(exits.index, exits['spread'], 
                          color='black', marker='x', s=50, label='Exit')
        
        axes[0, 1].set_title('Spread and Trading Signals')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Z-score
        if 'zscore' in self.signals_data.columns:
            axes[1, 0].plot(self.signals_data['zscore'], alpha=0.7)
            axes[1, 0].axhline(y=self.trading_params['entry_threshold'], color='red', linestyle='--', alpha=0.5)
            axes[1, 0].axhline(y=-self.trading_params['entry_threshold'], color='red', linestyle='--', alpha=0.5)
            axes[1, 0].axhline(y=self.trading_params['exit_threshold'], color='green', linestyle=':', alpha=0.5)
            axes[1, 0].axhline(y=-self.trading_params['exit_threshold'], color='green', linestyle=':', alpha=0.5)
            axes[1, 0].set_title('Spread Z-Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Positions
        axes[1, 1].plot(self.signals_data['position'], drawstyle='steps-post')
        axes[1, 1].set_title('Position')
        axes[1, 1].set_ylabel('Position Size')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Cumulative returns
        if 'cumulative_return' in self.signals_data.columns:
            axes[2, 0].plot((1 + self.signals_data['cumulative_return']) * 100, label='Strategy')
            axes[2, 0].set_title('Cumulative Returns (%)')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Performance summary
        axes[2, 1].axis('off')
        if self.performance_metrics:
            perf_text = f"""
Performance Summary:
Total Return: {self.performance_metrics['total_return']*100:.2f}%
Annual Return: {self.performance_metrics['annualized_return']*100:.2f}%
Volatility: {self.performance_metrics['volatility']*100:.2f}%
Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}
Max Drawdown: {self.performance_metrics['max_drawdown']*100:.2f}%
Number of Trades: {self.performance_metrics['number_of_trades']}
Win Rate: {self.performance_metrics['win_rate']*100:.1f}%
Profit Factor: {self.performance_metrics['profit_factor']:.2f}

Pair Statistics:
Hedge Ratio: {self.hedge_ratio:.4f}
Cointegrated: {self.cointegration_results['is_cointegrated']}
P-value: {self.cointegration_results['min_p_value']:.6f}
            """
            axes[2, 1].text(0.1, 0.9, perf_text, transform=axes[2, 1].transAxes,
                            verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Strategy analysis plot saved to: {save_path}")
        
        plt.show()
    
    def get_strategy_report(self) -> str:
        """
        Generate comprehensive strategy analysis report
        
        Returns:
        --------
        str containing formatted strategy report
        """
        if not all([
            self.cointegration_results is not None,
            self.signals_data is not None and not self.signals_data.empty,
            self.performance_metrics is not None
        ]):
            return "Incomplete strategy analysis. Run the full analysis pipeline first."
        
        report = []
        report.append("=" * 80)
        report.append(f"PAIRS TRADING STRATEGY REPORT: {self.asset1} vs {self.asset2}")
        report.append("=" * 80)
        
        # Basic information
        report.append(f"Analysis Period: {self.price_data.index[0]} to {self.price_data.index[-1]}")
        report.append(f"Total Observations: {len(self.price_data)}")
        report.append(f"Lookback Period: {self.lookback_period} days")
        report.append("")
        
        # Cointegration analysis
        report.append("COINTEGRATION ANALYSIS")
        report.append("-" * 30)
        report.append(f"Is Cointegrated: {self.cointegration_results['is_cointegrated']}")
        report.append(f"P-value: {self.cointegration_results['min_p_value']:.6f}")
        report.append(f"Hedge Ratio: {self.hedge_ratio:.4f}")
        
        if 'spread_analysis' in self.cointegration_results:
            spread_analysis = self.cointegration_results['spread_analysis']
            if 'half_life' in spread_analysis and 'error' not in spread_analysis['half_life']:
                report.append(f"Mean Reversion Half-life: {spread_analysis['half_life']['days']:.1f} days")
                report.append(f"Half-life Suitable: {spread_analysis['half_life']['is_suitable']}")
        
        report.append("")
        
        # Trading parameters
        report.append("TRADING PARAMETERS")
        report.append("-" * 20)
        for key, value in self.trading_params.items():
            report.append(f"{key}: {value}")
        report.append("")
        
        # Performance metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 20)
        perf = self.performance_metrics
        report.append(f"Total Return: {perf['total_return']*100:.2f}%")
        report.append(f"Annualized Return: {perf['annualized_return']*100:.2f}%")
        report.append(f"Volatility: {perf['volatility']*100:.2f}%")
        report.append(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        report.append(f"Information Ratio: {perf['information_ratio']:.2f}")
        report.append(f"Calmar Ratio: {perf['calmar_ratio']:.2f}")
        report.append(f"Maximum Drawdown: {perf['max_drawdown']*100:.2f}%")
        report.append(f"Max DD Duration: {perf['max_drawdown_duration_days']} days")
        report.append(f"VaR (95%): {perf['var_95']*100:.2f}%")
        report.append(f"CVaR (95%): {perf['cvar_95']*100:.2f}%")
        report.append("")
        
        # Trade statistics
        report.append("TRADE STATISTICS")
        report.append("-" * 17)
        report.append(f"Number of Trades: {perf['number_of_trades']}")
        report.append(f"Win Rate: {perf['win_rate']*100:.1f}%")
        report.append(f"Average Win: {perf['avg_win']*100:.2f}%")
        report.append(f"Average Loss: {perf['avg_loss']*100:.2f}%")
        report.append(f"Profit Factor: {perf['profit_factor']:.2f}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
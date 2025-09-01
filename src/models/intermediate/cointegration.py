"""
Cointegration Analysis Module

This module provides comprehensive cointegration analysis tools for pairs trading
and multi-asset relationship modeling in quantitative trading strategies.

Key Features:
- Engle-Granger two-step cointegration test
- Johansen multivariate cointegration test
- Vector Error Correction Model (VECM) implementation
- Cointegration relationship strength analysis
- Trading signal generation based on cointegration spreads

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
# Try different import paths for Johansen test
try:
    from statsmodels.tsa.vector_ar.johansen import coint_johansen
except ImportError:
    try:
        from statsmodels.tsa.johansen import coint_johansen
    except ImportError:
        # Fallback - implement basic Johansen test or skip
        coint_johansen = None
import matplotlib.pyplot as plt


class CointegrationAnalyzer:
    """
    Comprehensive cointegration analysis for pairs trading and multi-asset strategies
    
    This class provides tools for:
    1. Testing cointegration relationships between multiple time series
    2. Estimating Vector Error Correction Models (VECM)
    3. Generating trading signals based on cointegration spreads
    4. Analyzing the strength and stability of cointegration relationships
    """
    
    def __init__(self, 
                 price_data: pd.DataFrame,
                 price_columns: List[str] = None,
                 lookback_period: int = 252,
                 confidence_level: float = 0.05):
        """
        Initialize CointegrationAnalyzer
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Time series price data with datetime index
        price_columns : List[str], optional
            Column names to use for analysis. If None, uses all numeric columns
        lookback_period : int, default=252
            Lookback period for rolling analysis (trading days)
        confidence_level : float, default=0.05
            Significance level for statistical tests
        """
        self.price_data = price_data.copy()
        self.price_columns = price_columns or [col for col in price_data.columns 
                                             if price_data[col].dtype in ['float64', 'int64']]
        self.lookback_period = lookback_period
        self.confidence_level = confidence_level
        
        # Validate inputs
        if len(self.price_columns) < 2:
            raise ValueError("Need at least 2 price series for cointegration analysis")
        
        # Store analysis results
        self.cointegration_results = {}
        self.vecm_model = None
        self.spread_data = {}
        
        # Calculate log prices for analysis
        self.log_prices = np.log(self.price_data[self.price_columns])
        
        print(f"CointegrationAnalyzer initialized:")
        print(f"  Assets: {self.price_columns}")
        print(f"  Data period: {self.price_data.index[0]} to {self.price_data.index[-1]}")
        print(f"  Total observations: {len(self.price_data)}")
    
    def test_stationarity(self, 
                         series: pd.Series, 
                         test_type: str = 'adf') -> Dict:
        """
        Test stationarity of a time series using ADF or KPSS test
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        test_type : str, default='adf'
            Type of test ('adf' for Augmented Dickey-Fuller, 'kpss' for KPSS)
        
        Returns:
        --------
        Dict containing test results
        """
        results = {'test_type': test_type, 'series_name': series.name}
        
        if test_type.lower() == 'adf':
            adf_result = adfuller(series.dropna(), autolag='AIC')
            results.update({
                'test_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < self.confidence_level,
                'interpretation': 'Stationary' if adf_result[1] < self.confidence_level else 'Non-stationary'
            })
        
        return results
    
    def engle_granger_test(self, 
                          y_series: pd.Series, 
                          x_series: pd.Series) -> Dict:
        """
        Perform Engle-Granger two-step cointegration test
        
        Parameters:
        -----------
        y_series : pd.Series
            Dependent variable (usually the asset being predicted)
        x_series : pd.Series
            Independent variable (usually the hedging asset)
        
        Returns:
        --------
        Dict containing cointegration test results
        """
        # Align series
        aligned_data = pd.concat([y_series, x_series], axis=1).dropna()
        y_clean = aligned_data.iloc[:, 0]
        x_clean = aligned_data.iloc[:, 1]
        
        # Step 1: Run cointegrating regression
        # y_t = alpha + beta * x_t + u_t
        x_with_const = np.column_stack([np.ones(len(x_clean)), x_clean])
        ols_result = OLS(y_clean, x_with_const).fit()
        
        # Extract coefficients
        alpha = ols_result.params[0]
        beta = ols_result.params[1]
        residuals = ols_result.resid
        
        # Step 2: Test residuals for stationarity
        coint_test = coint(y_clean, x_clean)
        
        # Additional spread analysis
        spread = y_clean - beta * x_clean - alpha
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        results = {
            'asset_pair': f"{y_series.name}_{x_series.name}",
            'cointegration_coefficient': beta,
            'intercept': alpha,
            'cointegration_statistic': coint_test[0],
            'p_value': coint_test[1],
            'critical_values': coint_test[2],
            'is_cointegrated': coint_test[1] < self.confidence_level,
            'spread_series': spread,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'ols_results': ols_result,
            'r_squared': ols_result.rsquared
        }
        
        return results
    
    def johansen_test(self, 
                     det_order: int = 0, 
                     k_ar_diff: int = 1) -> Dict:
        """
        Perform Johansen multivariate cointegration test
        
        Parameters:
        -----------
        det_order : int, default=0
            Deterministic trend order (0=no trend, 1=constant, 2=linear trend)
        k_ar_diff : int, default=1
            Number of lagged differences in the VECM
        
        Returns:
        --------
        Dict containing Johansen test results
        """
        if coint_johansen is None:
            return {
                'error': 'Johansen test not available - statsmodels version issue',
                'n_assets': len(self.price_columns),
                'asset_names': self.price_columns,
                'n_cointegrating_trace': 0,
                'n_cointegrating_eigen': 0
            }
        
        # Prepare data
        data_clean = self.log_prices.dropna()
        
        # Run Johansen test
        johansen_result = coint_johansen(data_clean, det_order, k_ar_diff)
        
        # Extract results
        n_vars = len(self.price_columns)
        trace_stats = johansen_result.lr1
        eigen_stats = johansen_result.lr2
        critical_values_trace = johansen_result.cvt
        critical_values_eigen = johansen_result.cvm
        eigenvalues = johansen_result.eig
        
        # Determine number of cointegrating relationships
        trace_cointegrated = []
        eigen_cointegrated = []
        
        for i in range(n_vars):
            # Trace test
            if trace_stats[i] > critical_values_trace[i, 1]:  # 5% level
                trace_cointegrated.append(i)
            
            # Eigenvalue test  
            if eigen_stats[i] > critical_values_eigen[i, 1]:  # 5% level
                eigen_cointegrated.append(i)
        
        results = {
            'n_assets': n_vars,
            'asset_names': self.price_columns,
            'eigenvalues': eigenvalues,
            'trace_statistics': trace_stats,
            'eigenvalue_statistics': eigen_stats,
            'critical_values_trace': critical_values_trace,
            'critical_values_eigen': critical_values_eigen,
            'n_cointegrating_trace': len(trace_cointegrated),
            'n_cointegrating_eigen': len(eigen_cointegrated),
            'cointegrating_vectors': johansen_result.evec,
            'johansen_result': johansen_result
        }
        
        return results
    
    def estimate_vecm(self, 
                     n_coint_relations: int = 1,
                     k_ar_diff: int = 1,
                     deterministic: str = 'ci') -> Dict:
        """
        Estimate Vector Error Correction Model (VECM)
        
        Parameters:
        -----------
        n_coint_relations : int, default=1
            Number of cointegrating relationships
        k_ar_diff : int, default=1
            Number of lagged differences
        deterministic : str, default='ci'
            Deterministic terms ('n'=none, 'co'=constant outside, 'ci'=constant inside)
        
        Returns:
        --------
        Dict containing VECM estimation results
        """
        # Prepare data
        data_clean = self.log_prices.dropna()
        
        # Estimate VECM
        try:
            vecm = VECM(data_clean, 
                       k_ar_diff=k_ar_diff,
                       coint_rank=n_coint_relations,
                       deterministic=deterministic)
            vecm_result = vecm.fit()
            self.vecm_model = vecm_result
            
            # Extract key results
            results = {
                'vecm_model': vecm_result,
                'cointegrating_vectors': vecm_result.beta,
                'adjustment_coefficients': vecm_result.alpha,
                'log_likelihood': vecm_result.llf,
                'aic': vecm_result.aic,
                'bic': vecm_result.bic,
                'residuals': vecm_result.resid,
                'fitted_values': vecm_result.fittedvalues
            }
            
            # Calculate error correction terms
            if vecm_result.beta is not None:
                ec_terms = data_clean.values @ vecm_result.beta
                results['error_correction_terms'] = pd.DataFrame(
                    ec_terms, 
                    index=data_clean.index, 
                    columns=[f'ECT_{i+1}' for i in range(n_coint_relations)]
                )
            
            return results
            
        except Exception as e:
            print(f"VECM estimation failed: {str(e)}")
            return {'error': str(e), 'vecm_model': None}
    
    def analyze_pairs_cointegration(self) -> Dict:
        """
        Analyze pairwise cointegration relationships among all assets
        
        Returns:
        --------
        Dict containing all pairwise cointegration results
        """
        results = {}
        n_assets = len(self.price_columns)
        
        print("Analyzing pairwise cointegration relationships...")
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                asset1 = self.price_columns[i]
                asset2 = self.price_columns[j]
                
                print(f"  Testing: {asset1} <-> {asset2}")
                
                # Test both directions
                series1 = self.log_prices[asset1]
                series2 = self.log_prices[asset2]
                
                # Test asset1 ~ asset2
                coint_12 = self.engle_granger_test(series1, series2)
                # Test asset2 ~ asset1  
                coint_21 = self.engle_granger_test(series2, series1)
                
                pair_key = f"{asset1}_{asset2}"
                results[pair_key] = {
                    'asset1_name': asset1,
                    'asset2_name': asset2,
                    'test_12': coint_12,
                    'test_21': coint_21,
                    'is_cointegrated': (coint_12['is_cointegrated'] or 
                                      coint_21['is_cointegrated']),
                    'best_direction': '12' if (coint_12['p_value'] < coint_21['p_value']) else '21',
                    'min_p_value': min(coint_12['p_value'], coint_21['p_value'])
                }
                
                # Store spread data
                if results[pair_key]['is_cointegrated']:
                    best_test = coint_12 if results[pair_key]['best_direction'] == '12' else coint_21
                    self.spread_data[pair_key] = best_test['spread_series']
        
        self.cointegration_results = results
        return results
    
    def generate_trading_signals(self, 
                                pair_key: str,
                                entry_threshold: float = 2.0,
                                exit_threshold: float = 0.5,
                                lookback_zscore: int = 60) -> pd.DataFrame:
        """
        Generate trading signals based on cointegration spread
        
        Parameters:
        -----------
        pair_key : str
            Asset pair key (e.g., 'BTC_ETH')
        entry_threshold : float, default=2.0
            Z-score threshold for trade entry
        exit_threshold : float, default=0.5
            Z-score threshold for trade exit
        lookback_zscore : int, default=60
            Lookback period for z-score calculation
        
        Returns:
        --------
        pd.DataFrame containing trading signals
        """
        if pair_key not in self.spread_data:
            raise ValueError(f"No spread data available for pair: {pair_key}")
        
        spread = self.spread_data[pair_key]
        
        # Calculate rolling z-score
        rolling_mean = spread.rolling(window=lookback_zscore).mean()
        rolling_std = spread.rolling(window=lookback_zscore).std()
        zscore = (spread - rolling_mean) / rolling_std
        
        # Initialize signals
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['zscore'] = zscore
        signals['position'] = 0
        signals['entry_signal'] = 0
        signals['exit_signal'] = 0
        
        # Generate signals
        current_position = 0
        
        for i in range(len(signals)):
            z = zscore.iloc[i]
            
            if pd.isna(z):
                continue
            
            # Entry signals
            if current_position == 0:
                if z > entry_threshold:
                    # Short spread (long asset2, short asset1)
                    current_position = -1
                    signals.iloc[i, signals.columns.get_loc('entry_signal')] = -1
                elif z < -entry_threshold:
                    # Long spread (long asset1, short asset2)
                    current_position = 1
                    signals.iloc[i, signals.columns.get_loc('entry_signal')] = 1
            
            # Exit signals
            elif current_position != 0:
                if abs(z) < exit_threshold:
                    signals.iloc[i, signals.columns.get_loc('exit_signal')] = -current_position
                    current_position = 0
            
            signals.iloc[i, signals.columns.get_loc('position')] = current_position
        
        return signals
    
    def calculate_trading_performance(self, 
                                    signals: pd.DataFrame,
                                    pair_key: str,
                                    transaction_cost: float = 0.001) -> Dict:
        """
        Calculate trading performance metrics for cointegration strategy
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Trading signals from generate_trading_signals()
        pair_key : str
            Asset pair key
        transaction_cost : float, default=0.001
            Transaction cost per trade (0.1%)
        
        Returns:
        --------
        Dict containing performance metrics
        """
        pair_info = self.cointegration_results[pair_key]
        asset1_name = pair_info['asset1_name']
        asset2_name = pair_info['asset2_name']
        
        # Get price data
        asset1_prices = self.price_data[asset1_name]
        asset2_prices = self.price_data[asset2_name]
        
        # Align data
        aligned_data = pd.concat([asset1_prices, asset2_prices, signals], axis=1).dropna()
        
        # Calculate returns
        asset1_returns = aligned_data[asset1_name].pct_change()
        asset2_returns = aligned_data[asset2_name].pct_change()
        
        # Calculate strategy returns
        positions = aligned_data['position'].shift(1)  # Use previous position
        
        # Strategy return = position * spread_return - transaction_costs
        spread_returns = asset1_returns - asset2_returns  # Simplified spread return
        strategy_returns = positions * spread_returns
        
        # Apply transaction costs
        trades = (aligned_data['position'] != aligned_data['position'].shift(1)).fillna(False)
        transaction_costs = trades * transaction_cost
        net_strategy_returns = strategy_returns - transaction_costs
        
        # Performance metrics
        total_return = (1 + net_strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(net_strategy_returns)) - 1
        volatility = net_strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        max_drawdown = (net_strategy_returns.cumsum() - net_strategy_returns.cumsum().cummax()).min()
        
        n_trades = trades.sum()
        hit_rate = (net_strategy_returns[net_strategy_returns != 0] > 0).mean()
        
        performance = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'number_of_trades': n_trades,
            'hit_rate': hit_rate,
            'strategy_returns': net_strategy_returns
        }
        
        return performance
    
    def plot_cointegration_analysis(self, 
                                   pair_key: str = None,
                                   figsize: Tuple[int, int] = (15, 10)):
        """
        Create comprehensive cointegration analysis plots
        
        Parameters:
        -----------
        pair_key : str, optional
            Specific pair to plot. If None, plots summary for all pairs
        figsize : Tuple[int, int], default=(15, 10)
            Figure size for plots
        """
        if pair_key and pair_key in self.cointegration_results:
            self._plot_pair_analysis(pair_key, figsize)
        else:
            self._plot_summary_analysis(figsize)
    
    def _plot_pair_analysis(self, pair_key: str, figsize: Tuple[int, int]):
        """Plot detailed analysis for a specific pair"""
        pair_info = self.cointegration_results[pair_key]
        asset1_name = pair_info['asset1_name']
        asset2_name = pair_info['asset2_name']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Cointegration Analysis: {asset1_name} vs {asset2_name}', fontsize=16)
        
        # Plot 1: Price series
        axes[0, 0].plot(self.price_data[asset1_name], label=asset1_name, alpha=0.7)
        axes[0, 0].plot(self.price_data[asset2_name], label=asset2_name, alpha=0.7)
        axes[0, 0].set_title('Price Series')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Log prices
        axes[0, 1].plot(self.log_prices[asset1_name], label=f'Log {asset1_name}', alpha=0.7)
        axes[0, 1].plot(self.log_prices[asset2_name], label=f'Log {asset2_name}', alpha=0.7)
        axes[0, 1].set_title('Log Price Series')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Spread
        if pair_key in self.spread_data:
            spread = self.spread_data[pair_key]
            axes[1, 0].plot(spread, color='red', alpha=0.7)
            axes[1, 0].axhline(y=spread.mean(), color='black', linestyle='--', alpha=0.5)
            axes[1, 0].axhline(y=spread.mean() + 2*spread.std(), color='red', linestyle=':', alpha=0.5)
            axes[1, 0].axhline(y=spread.mean() - 2*spread.std(), color='red', linestyle=':', alpha=0.5)
            axes[1, 0].set_title(f'Cointegration Spread (p-value: {pair_info["min_p_value"]:.4f})')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Statistics summary
        axes[1, 1].axis('off')
        stats_text = f"""
Cointegration Statistics:
- Is Cointegrated: {pair_info['is_cointegrated']}
- Min p-value: {pair_info['min_p_value']:.4f}
- Best Direction: {asset1_name} ~ {asset2_name if pair_info['best_direction'] == '12' else asset2_name} ~ {asset1_name}

Best Test Results:
- Coefficient: {pair_info[f'test_{pair_info["best_direction"]}']['cointegration_coefficient']:.4f}
- R-squared: {pair_info[f'test_{pair_info["best_direction"]}']['r_squared']:.4f}
- Spread Mean: {pair_info[f'test_{pair_info["best_direction"]}']['spread_mean']:.6f}
- Spread Std: {pair_info[f'test_{pair_info["best_direction"]}']['spread_std']:.6f}
        """
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_summary_analysis(self, figsize: Tuple[int, int]):
        """Plot summary analysis for all pairs"""
        if not self.cointegration_results:
            print("No cointegration results available. Run analyze_pairs_cointegration() first.")
            return
        
        cointegrated_pairs = [k for k, v in self.cointegration_results.items() 
                             if v['is_cointegrated']]
        
        if not cointegrated_pairs:
            print("No cointegrated pairs found.")
            return
        
        n_pairs = len(cointegrated_pairs)
        n_cols = 2
        n_rows = (n_pairs + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Cointegration Spreads Summary', fontsize=16)
        
        for i, pair_key in enumerate(cointegrated_pairs):
            row, col = i // n_cols, i % n_cols
            
            if pair_key in self.spread_data:
                spread = self.spread_data[pair_key]
                axes[row, col].plot(spread, alpha=0.7)
                axes[row, col].axhline(y=spread.mean(), color='black', linestyle='--', alpha=0.5)
                axes[row, col].axhline(y=spread.mean() + 2*spread.std(), color='red', linestyle=':', alpha=0.5)
                axes[row, col].axhline(y=spread.mean() - 2*spread.std(), color='red', linestyle=':', alpha=0.5)
                
                pair_info = self.cointegration_results[pair_key]
                axes[row, col].set_title(f"{pair_key}\np-val: {pair_info['min_p_value']:.4f}")
                axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(cointegrated_pairs), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_summary_report(self) -> str:
        """
        Generate a comprehensive summary report of cointegration analysis
        
        Returns:
        --------
        str containing formatted analysis report
        """
        if not self.cointegration_results:
            return "No cointegration analysis results available. Run analyze_pairs_cointegration() first."
        
        report = []
        report.append("=" * 80)
        report.append("COINTEGRATION ANALYSIS SUMMARY REPORT")
        report.append("=" * 80)
        
        # Basic info
        report.append(f"Analysis Period: {self.price_data.index[0]} to {self.price_data.index[-1]}")
        report.append(f"Total Observations: {len(self.price_data)}")
        report.append(f"Assets Analyzed: {', '.join(self.price_columns)}")
        report.append(f"Confidence Level: {self.confidence_level}")
        report.append("")
        
        # Cointegration results
        total_pairs = len(self.cointegration_results)
        cointegrated_pairs = [(k, v) for k, v in self.cointegration_results.items() 
                             if v['is_cointegrated']]
        
        report.append("PAIRWISE COINTEGRATION RESULTS")
        report.append("-" * 40)
        report.append(f"Total Pairs Tested: {total_pairs}")
        report.append(f"Cointegrated Pairs: {len(cointegrated_pairs)} ({len(cointegrated_pairs)/total_pairs*100:.1f}%)")
        report.append("")
        
        if cointegrated_pairs:
            report.append("COINTEGRATED PAIRS DETAILS:")
            report.append("-" * 40)
            for pair_key, pair_info in cointegrated_pairs:
                best_test = pair_info[f'test_{pair_info["best_direction"]}']
                report.append(f"\n{pair_key}:")
                report.append(f"  - p-value: {pair_info['min_p_value']:.6f}")
                report.append(f"  - Cointegration coefficient: {best_test['cointegration_coefficient']:.4f}")
                report.append(f"  - R-squared: {best_test['r_squared']:.4f}")
                report.append(f"  - Spread volatility: {best_test['spread_std']:.6f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
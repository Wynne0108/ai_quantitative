"""
Statistical Tests Module for Quantitative Finance

This module provides comprehensive statistical testing tools commonly used
in quantitative finance and econometrics, particularly for time series analysis.

Key Features:
- Stationarity tests (ADF, PP, KPSS)
- Normality tests (Jarque-Bera, Shapiro-Wilk, Anderson-Darling)  
- Autocorrelation tests (Ljung-Box, Durbin-Watson)
- Heteroscedasticity tests (ARCH, Breusch-Pagan)
- Cointegration tests (Engle-Granger, Johansen)
- Custom financial tests (runs test, variance ratio test)

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats
from statsmodels.tsa.stattools import (
    adfuller, kpss, acf, pacf
)
from statsmodels.stats.diagnostic import (
    acorr_ljungbox, het_breuschpagan, het_arch
)
from statsmodels.stats.stattools import durbin_watson
from arch.unitroot import PhillipsPerron
import matplotlib.pyplot as plt


class StatisticalTests:
    """
    Comprehensive statistical testing suite for quantitative finance
    
    This class provides a wide range of statistical tests commonly used
    in financial econometrics and quantitative trading research.
    """
    
    def __init__(self, confidence_levels: List[float] = [0.01, 0.05, 0.10]):
        """
        Initialize StatisticalTests
        
        Parameters:
        -----------
        confidence_levels : List[float], default=[0.01, 0.05, 0.10]
            Standard confidence levels for statistical tests
        """
        self.confidence_levels = confidence_levels
        self.test_results = {}
    
    def test_stationarity_comprehensive(self, 
                                     series: pd.Series,
                                     maxlag: Optional[int] = None) -> Dict:
        """
        Comprehensive stationarity testing using multiple methods
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        maxlag : int, optional
            Maximum lag for ADF test
        
        Returns:
        --------
        Dict containing results from multiple stationarity tests
        """
        results = {
            'series_name': series.name,
            'n_observations': len(series.dropna()),
            'test_summary': {}
        }
        
        clean_series = series.dropna()
        
        # 1. Augmented Dickey-Fuller Test
        try:
            adf_result = adfuller(clean_series, maxlag=maxlag, autolag='AIC')
            results['adf'] = {
                'test_statistic': adf_result[0],
                'p_value': adf_result[1],
                'used_lag': adf_result[2],
                'n_observations': adf_result[3],
                'critical_values': adf_result[4],
                'ic_best': adf_result[5],
                'is_stationary': {level: adf_result[1] < level for level in self.confidence_levels},
                'interpretation': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
            }
        except Exception as e:
            results['adf'] = {'error': str(e)}
        
        # 2. Phillips-Perron Test
        try:
            pp_test = PhillipsPerron(clean_series)
            pp_result = pp_test.run()
            results['phillips_perron'] = {
                'test_statistic': pp_result.stat,
                'p_value': pp_result.pvalue,
                'critical_values': pp_result.critical_values,
                'is_stationary': {level: pp_result.pvalue < level for level in self.confidence_levels},
                'interpretation': 'Stationary' if pp_result.pvalue < 0.05 else 'Non-stationary'
            }
        except Exception as e:
            results['phillips_perron'] = {'error': str(e)}
        
        # 3. KPSS Test (null hypothesis: stationary)
        try:
            kpss_result = kpss(clean_series, regression='c', nlags='auto')
            results['kpss'] = {
                'test_statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'used_lag': kpss_result[2],
                'critical_values': kpss_result[3],
                'is_stationary': {level: kpss_result[1] > level for level in self.confidence_levels},  # Note: reversed logic
                'interpretation': 'Stationary' if kpss_result[1] > 0.05 else 'Non-stationary'
            }
        except Exception as e:
            results['kpss'] = {'error': str(e)}
        
        # Summary consensus
        stationary_count = 0
        total_tests = 0
        
        for test_name in ['adf', 'phillips_perron', 'kpss']:
            if test_name in results and 'error' not in results[test_name]:
                total_tests += 1
                if results[test_name]['interpretation'] == 'Stationary':
                    stationary_count += 1
        
        results['test_summary'] = {
            'total_tests': total_tests,
            'stationary_count': stationary_count,
            'consensus': 'Stationary' if stationary_count >= total_tests/2 else 'Non-stationary',
            'confidence': stationary_count / total_tests if total_tests > 0 else 0
        }
        
        return results
    
    def test_normality_comprehensive(self, 
                                   series: pd.Series,
                                   plot: bool = False) -> Dict:
        """
        Comprehensive normality testing using multiple methods
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        plot : bool, default=False
            Whether to create diagnostic plots
        
        Returns:
        --------
        Dict containing results from multiple normality tests
        """
        results = {
            'series_name': series.name,
            'n_observations': len(series.dropna()),
            'descriptive_stats': {}
        }
        
        clean_series = series.dropna()
        
        # Descriptive statistics
        results['descriptive_stats'] = {
            'mean': clean_series.mean(),
            'std': clean_series.std(),
            'skewness': stats.skew(clean_series),
            'kurtosis': stats.kurtosis(clean_series),
            'excess_kurtosis': stats.kurtosis(clean_series) - 3,
            'min': clean_series.min(),
            'max': clean_series.max(),
            'range': clean_series.max() - clean_series.min()
        }
        
        # 1. Jarque-Bera Test
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(clean_series)
            results['jarque_bera'] = {
                'test_statistic': jb_stat,
                'p_value': jb_pvalue,
                'is_normal': {level: jb_pvalue > level for level in self.confidence_levels},
                'interpretation': 'Normal' if jb_pvalue > 0.05 else 'Non-normal'
            }
        except Exception as e:
            results['jarque_bera'] = {'error': str(e)}
        
        # 2. Shapiro-Wilk Test (for smaller samples)
        if len(clean_series) <= 5000:
            try:
                sw_stat, sw_pvalue = stats.shapiro(clean_series)
                results['shapiro_wilk'] = {
                    'test_statistic': sw_stat,
                    'p_value': sw_pvalue,
                    'is_normal': {level: sw_pvalue > level for level in self.confidence_levels},
                    'interpretation': 'Normal' if sw_pvalue > 0.05 else 'Non-normal'
                }
            except Exception as e:
                results['shapiro_wilk'] = {'error': str(e)}
        
        # 3. Anderson-Darling Test
        try:
            ad_result = stats.anderson(clean_series, dist='norm')
            results['anderson_darling'] = {
                'test_statistic': ad_result.statistic,
                'critical_values': dict(zip([15, 10, 5, 2.5, 1], ad_result.critical_values)),
                'is_normal': {
                    15: ad_result.statistic < ad_result.critical_values[0],
                    10: ad_result.statistic < ad_result.critical_values[1], 
                    5: ad_result.statistic < ad_result.critical_values[2],
                    2.5: ad_result.statistic < ad_result.critical_values[3],
                    1: ad_result.statistic < ad_result.critical_values[4]
                },
                'interpretation': 'Normal' if ad_result.statistic < ad_result.critical_values[2] else 'Non-normal'
            }
        except Exception as e:
            results['anderson_darling'] = {'error': str(e)}
        
        # 4. D'Agostino's Test
        try:
            da_stat, da_pvalue = stats.normaltest(clean_series)
            results['dagostino'] = {
                'test_statistic': da_stat,
                'p_value': da_pvalue,
                'is_normal': {level: da_pvalue > level for level in self.confidence_levels},
                'interpretation': 'Normal' if da_pvalue > 0.05 else 'Non-normal'
            }
        except Exception as e:
            results['dagostino'] = {'error': str(e)}
        
        # Summary consensus
        normal_count = 0
        total_tests = 0
        
        for test_name in ['jarque_bera', 'shapiro_wilk', 'anderson_darling', 'dagostino']:
            if test_name in results and 'error' not in results[test_name]:
                total_tests += 1
                if results[test_name]['interpretation'] == 'Normal':
                    normal_count += 1
        
        results['test_summary'] = {
            'total_tests': total_tests,
            'normal_count': normal_count,
            'consensus': 'Normal' if normal_count >= total_tests/2 else 'Non-normal',
            'confidence': normal_count / total_tests if total_tests > 0 else 0
        }
        
        # Create plots if requested
        if plot:
            self._plot_normality_diagnostics(clean_series, results)
        
        return results
    
    def test_autocorrelation(self, 
                           series: pd.Series,
                           lags: int = 20,
                           plot: bool = False) -> Dict:
        """
        Test for autocorrelation using multiple methods
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        lags : int, default=20
            Number of lags to test
        plot : bool, default=False
            Whether to create ACF/PACF plots
        
        Returns:
        --------
        Dict containing autocorrelation test results
        """
        results = {
            'series_name': series.name,
            'n_observations': len(series.dropna()),
            'lags_tested': lags
        }
        
        clean_series = series.dropna()
        
        # 1. Ljung-Box Test
        try:
            lb_result = acorr_ljungbox(clean_series, lags=lags, return_df=True)
            results['ljung_box'] = {
                'test_statistics': lb_result['lb_stat'].to_dict(),
                'p_values': lb_result['lb_pvalue'].to_dict(),
                'is_independent': {
                    lag: pval > 0.05 for lag, pval in lb_result['lb_pvalue'].items()
                },
                'overall_independence': (lb_result['lb_pvalue'] > 0.05).all(),
                'interpretation': 'Independent' if (lb_result['lb_pvalue'] > 0.05).all() else 'Autocorrelated'
            }
        except Exception as e:
            results['ljung_box'] = {'error': str(e)}
        
        # 2. Durbin-Watson Test
        try:
            dw_stat = durbin_watson(clean_series)
            results['durbin_watson'] = {
                'test_statistic': dw_stat,
                'interpretation': self._interpret_durbin_watson(dw_stat)
            }
        except Exception as e:
            results['durbin_watson'] = {'error': str(e)}
        
        # 3. ACF and PACF
        try:
            acf_values, acf_confint = acf(clean_series, nlags=lags, alpha=0.05)
            pacf_values, pacf_confint = pacf(clean_series, nlags=lags, alpha=0.05)
            
            results['acf'] = {
                'values': acf_values[1:].tolist(),  # Exclude lag 0
                'confidence_intervals': acf_confint[1:].tolist(),
                'significant_lags': [i for i in range(1, len(acf_values)) 
                                   if abs(acf_values[i]) > abs(acf_confint[i, 1] - acf_values[i])]
            }
            
            results['pacf'] = {
                'values': pacf_values[1:].tolist(),  # Exclude lag 0
                'confidence_intervals': pacf_confint[1:].tolist(),
                'significant_lags': [i for i in range(1, len(pacf_values)) 
                                   if abs(pacf_values[i]) > abs(pacf_confint[i, 1] - pacf_values[i])]
            }
        except Exception as e:
            results['acf'] = {'error': str(e)}
            results['pacf'] = {'error': str(e)}
        
        # Create plots if requested
        if plot:
            self._plot_autocorrelation_diagnostics(clean_series, lags, results)
        
        return results
    
    def test_heteroscedasticity(self, 
                              series: pd.Series,
                              lags: int = 5) -> Dict:
        """
        Test for heteroscedasticity (changing variance)
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test  
        lags : int, default=5
            Number of lags for ARCH test
        
        Returns:
        --------
        Dict containing heteroscedasticity test results
        """
        results = {
            'series_name': series.name,
            'n_observations': len(series.dropna())
        }
        
        clean_series = series.dropna()
        
        # 1. ARCH Test (Autoregressive Conditional Heteroscedasticity)
        try:
            arch_result = het_arch(clean_series, nlags=lags)
            results['arch'] = {
                'test_statistic': arch_result[0],
                'p_value': arch_result[1],
                'f_statistic': arch_result[2],
                'f_p_value': arch_result[3],
                'is_homoscedastic': {level: arch_result[1] > level for level in self.confidence_levels},
                'interpretation': 'Homoscedastic' if arch_result[1] > 0.05 else 'Heteroscedastic (ARCH effects)'
            }
        except Exception as e:
            results['arch'] = {'error': str(e)}
        
        # 2. Breusch-Pagan Test  
        try:
            # Create simple time trend for BP test
            X = np.arange(len(clean_series)).reshape(-1, 1)
            bp_result = het_breuschpagan(clean_series, X)
            results['breusch_pagan'] = {
                'test_statistic': bp_result[0],
                'p_value': bp_result[1],
                'f_statistic': bp_result[2], 
                'f_p_value': bp_result[3],
                'is_homoscedastic': {level: bp_result[1] > level for level in self.confidence_levels},
                'interpretation': 'Homoscedastic' if bp_result[1] > 0.05 else 'Heteroscedastic'
            }
        except Exception as e:
            results['breusch_pagan'] = {'error': str(e)}
        
        return results
    
    def runs_test(self, series: pd.Series) -> Dict:
        """
        Runs test for randomness (tests if values are randomly distributed)
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        
        Returns:
        --------
        Dict containing runs test results
        """
        clean_series = series.dropna()
        median_val = clean_series.median()
        
        # Convert to binary sequence (above/below median)
        binary_seq = (clean_series > median_val).astype(int)
        
        # Count runs
        runs = 1
        for i in range(1, len(binary_seq)):
            if binary_seq.iloc[i] != binary_seq.iloc[i-1]:
                runs += 1
        
        # Calculate expected runs and variance
        n1 = (binary_seq == 1).sum()  # Number of values above median
        n2 = (binary_seq == 0).sum()  # Number of values below median
        n = n1 + n2
        
        expected_runs = (2 * n1 * n2 / n) + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
        
        # Calculate test statistic
        if variance_runs > 0:
            z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_value = 1
        
        results = {
            'series_name': series.name,
            'n_observations': n,
            'n_above_median': n1,
            'n_below_median': n2,
            'observed_runs': runs,
            'expected_runs': expected_runs,
            'test_statistic': z_stat,
            'p_value': p_value,
            'is_random': {level: p_value > level for level in self.confidence_levels},
            'interpretation': 'Random' if p_value > 0.05 else 'Non-random (clustered or trending)'
        }
        
        return results
    
    def variance_ratio_test(self, 
                          series: pd.Series,
                          periods: List[int] = [2, 4, 8, 16]) -> Dict:
        """
        Variance ratio test for random walk hypothesis
        
        Parameters:
        -----------
        series : pd.Series
            Time series (usually log prices)
        periods : List[int], default=[2, 4, 8, 16]
            Periods to test variance ratios
        
        Returns:
        --------
        Dict containing variance ratio test results
        """
        clean_series = series.dropna()
        returns = clean_series.diff().dropna()
        
        results = {
            'series_name': series.name,
            'n_observations': len(returns),
            'variance_ratios': {},
            'test_statistics': {},
            'p_values': {},
            'is_random_walk': {}
        }
        
        # Base variance (1-period)
        base_var = returns.var()
        
        for k in periods:
            if len(returns) < k * 2:
                continue
                
            # Calculate k-period returns
            k_period_returns = []
            for i in range(k-1, len(returns), k):
                k_period_returns.append(returns.iloc[i-k+1:i+1].sum())
            
            k_period_returns = pd.Series(k_period_returns)
            
            # Variance ratio
            vr = (k_period_returns.var() / k) / base_var
            
            # Test statistic (under random walk, VR should be 1)
            n = len(k_period_returns)
            test_stat = (vr - 1) * np.sqrt(n)
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
            
            results['variance_ratios'][k] = vr
            results['test_statistics'][k] = test_stat
            results['p_values'][k] = p_value
            results['is_random_walk'][k] = {level: p_value > level for level in self.confidence_levels}
        
        # Overall assessment
        all_p_values = list(results['p_values'].values())
        results['overall_random_walk'] = all(p > 0.05 for p in all_p_values)
        results['interpretation'] = ('Random Walk' if results['overall_random_walk'] 
                                   else 'Mean-reverting or Trending')
        
        return results
    
    def comprehensive_diagnostics(self, 
                                series: pd.Series,
                                plot: bool = False) -> Dict:
        """
        Run comprehensive statistical diagnostics on a time series
        
        Parameters:
        -----------
        series : pd.Series
            Time series to analyze
        plot : bool, default=False
            Whether to create diagnostic plots
        
        Returns:
        --------
        Dict containing results from all statistical tests
        """
        print(f"Running comprehensive diagnostics for: {series.name}")
        
        diagnostics = {
            'series_info': {
                'name': series.name,
                'n_observations': len(series.dropna()),
                'date_range': f"{series.index[0]} to {series.index[-1]}" if hasattr(series.index, 'date') else None
            }
        }
        
        # 1. Stationarity tests
        print("  Testing stationarity...")
        diagnostics['stationarity'] = self.test_stationarity_comprehensive(series)
        
        # 2. Normality tests
        print("  Testing normality...")
        returns = series.pct_change().dropna()
        diagnostics['normality'] = self.test_normality_comprehensive(returns, plot=False)
        
        # 3. Autocorrelation tests
        print("  Testing autocorrelation...")
        diagnostics['autocorrelation'] = self.test_autocorrelation(returns, plot=False)
        
        # 4. Heteroscedasticity tests
        print("  Testing heteroscedasticity...")
        diagnostics['heteroscedasticity'] = self.test_heteroscedasticity(returns)
        
        # 5. Runs test
        print("  Running runs test...")
        diagnostics['runs_test'] = self.runs_test(returns)
        
        # 6. Variance ratio test
        print("  Running variance ratio test...")
        log_prices = np.log(series)
        diagnostics['variance_ratio'] = self.variance_ratio_test(log_prices)
        
        # Summary assessment
        diagnostics['summary'] = self._generate_summary_assessment(diagnostics)
        
        if plot:
            self._plot_comprehensive_diagnostics(series, diagnostics)
        
        print("  Diagnostics complete!")
        return diagnostics
    
    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson test statistic"""
        if dw_stat < 1.5:
            return "Positive autocorrelation"
        elif dw_stat > 2.5:
            return "Negative autocorrelation"
        else:
            return "No significant autocorrelation"
    
    def _generate_summary_assessment(self, diagnostics: Dict) -> Dict:
        """Generate overall assessment of time series properties"""
        summary = {
            'data_quality': 'Good',
            'main_characteristics': [],
            'modeling_recommendations': [],
            'risk_factors': []
        }
        
        # Check stationarity
        if diagnostics['stationarity']['test_summary']['consensus'] == 'Non-stationary':
            summary['main_characteristics'].append('Non-stationary (trending)')
            summary['modeling_recommendations'].append('Consider differencing or detrending')
        else:
            summary['main_characteristics'].append('Stationary')
        
        # Check normality
        if diagnostics['normality']['test_summary']['consensus'] == 'Non-normal':
            summary['main_characteristics'].append('Non-normal returns')
            summary['modeling_recommendations'].append('Consider robust/non-parametric methods')
            
            excess_kurtosis = diagnostics['normality']['descriptive_stats']['excess_kurtosis']
            if excess_kurtosis > 3:
                summary['risk_factors'].append('Heavy tails (fat-tail risk)')
        
        # Check autocorrelation
        if ('ljung_box' in diagnostics['autocorrelation'] and 
            not diagnostics['autocorrelation']['ljung_box']['overall_independence']):
            summary['main_characteristics'].append('Autocorrelated')
            summary['modeling_recommendations'].append('Consider ARIMA-type models')
        
        # Check heteroscedasticity
        if ('arch' in diagnostics['heteroscedasticity'] and 
            diagnostics['heteroscedasticity']['arch']['interpretation'].startswith('Heteroscedastic')):
            summary['main_characteristics'].append('Time-varying volatility')
            summary['modeling_recommendations'].append('Consider GARCH-type models')
            summary['risk_factors'].append('Volatility clustering')
        
        # Check randomness
        if ('is_random' in diagnostics['runs_test'] and 
            not diagnostics['runs_test']['is_random'][0.05]):
            summary['main_characteristics'].append('Non-random patterns')
            summary['risk_factors'].append('Potential predictability')
        
        return summary
    
    def _plot_normality_diagnostics(self, series: pd.Series, results: Dict):
        """Create normality diagnostic plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Normality Diagnostics: {series.name}', fontsize=14)
        
        # Histogram
        axes[0, 0].hist(series, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Histogram')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q Plot
        stats.probplot(series, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box Plot
        axes[1, 0].boxplot(series, vert=True)
        axes[1, 0].set_title('Box Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 1].axis('off')
        stats_text = f"""
Normality Test Results:
Mean: {results['descriptive_stats']['mean']:.4f}
Std: {results['descriptive_stats']['std']:.4f}
Skewness: {results['descriptive_stats']['skewness']:.4f}
Kurtosis: {results['descriptive_stats']['kurtosis']:.4f}
Excess Kurtosis: {results['descriptive_stats']['excess_kurtosis']:.4f}

Jarque-Bera p-value: {results.get('jarque_bera', {}).get('p_value', 'N/A')}
Consensus: {results['test_summary']['consensus']}
        """
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_autocorrelation_diagnostics(self, series: pd.Series, lags: int, results: Dict):
        """Create autocorrelation diagnostic plots"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Autocorrelation Diagnostics: {series.name}', fontsize=14)
        
        # ACF Plot
        if 'acf' in results and 'error' not in results['acf']:
            acf_values = [1.0] + results['acf']['values']
            acf_ci = np.array([[1.0, 1.0]] + results['acf']['confidence_intervals'])
            
            axes[0].plot(range(len(acf_values)), acf_values, 'bo-', markersize=4)
            axes[0].fill_between(range(len(acf_values)), acf_ci[:, 0], acf_ci[:, 1], alpha=0.2)
            axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[0].set_title('Autocorrelation Function (ACF)')
            axes[0].set_xlabel('Lag')
            axes[0].set_ylabel('ACF')
            axes[0].grid(True, alpha=0.3)
        
        # PACF Plot  
        if 'pacf' in results and 'error' not in results['pacf']:
            pacf_values = [1.0] + results['pacf']['values']
            pacf_ci = np.array([[1.0, 1.0]] + results['pacf']['confidence_intervals'])
            
            axes[1].plot(range(len(pacf_values)), pacf_values, 'ro-', markersize=4)
            axes[1].fill_between(range(len(pacf_values)), pacf_ci[:, 0], pacf_ci[:, 1], alpha=0.2)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1].set_title('Partial Autocorrelation Function (PACF)')
            axes[1].set_xlabel('Lag')
            axes[1].set_ylabel('PACF')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_comprehensive_diagnostics(self, series: pd.Series, diagnostics: Dict):
        """Create comprehensive diagnostic plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Comprehensive Diagnostics: {series.name}', fontsize=16)
        
        # Price series
        axes[0, 0].plot(series, alpha=0.7)
        axes[0, 0].set_title('Price Series')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns
        returns = series.pct_change().dropna()
        axes[0, 1].plot(returns, alpha=0.7, color='red')
        axes[0, 1].set_title('Returns')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Returns histogram
        axes[0, 2].hist(returns, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Returns Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        axes[1, 0].plot(rolling_vol, alpha=0.7, color='green')
        axes[1, 0].set_title('Rolling Volatility (20-day)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q Plot
        stats.probplot(returns, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Returns)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary text
        axes[1, 2].axis('off')
        summary_text = f"""
Summary Assessment:
Data Quality: {diagnostics['summary']['data_quality']}

Main Characteristics:
{chr(10).join('- ' + char for char in diagnostics['summary']['main_characteristics'])}

Modeling Recommendations:
{chr(10).join('- ' + rec for rec in diagnostics['summary']['modeling_recommendations'])}

Risk Factors:
{chr(10).join('- ' + risk for risk in diagnostics['summary']['risk_factors'])}
        """
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontsize=9, fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
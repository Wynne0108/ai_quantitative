"""
Multi-Timeframe Cointegration Analysis

This script tests cointegration analysis across different time windows and parameters:
1. Multiple lookback periods (30, 60, 120, 252, 504 days)  
2. Different z-score calculation windows (30, 60, 120 periods)
3. Various entry/exit thresholds
4. Rolling cointegration stability testing
5. Comprehensive performance comparison

Usage: python scripts/test_multi_timeframe_cointegration.py

Author: Quantitative Trading Team
Date: 2025-09-02
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import itertools
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

# Import custom modules
from data_collection.binance_api import BinanceDataCollector
from models.intermediate.cointegration import CointegrationAnalyzer


def collect_multi_timeframe_data():
    """Collect data for different timeframes"""
    print("Collecting multi-timeframe cryptocurrency data...")
    
    try:
        collector = BinanceDataCollector()
        
        # Collect longer period data for comprehensive testing
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        
        # Try different intervals
        intervals_to_test = {
            '15m': 30,   # 30 days of 15min data
            '1h': 90,    # 90 days of hourly data  
            '4h': 180,   # 180 days of 4h data
            '1d': 365    # 365 days of daily data
        }
        
        collected_data = {}
        
        for interval, days in intervals_to_test.items():
            print(f"  Collecting {interval} data for {days} days...")
            
            interval_data = {}
            for symbol in symbols:
                try:
                    # Calculate start_time based on days back
                    end_time = datetime.now()
                    start_time = end_time - timedelta(days=days)
                    
                    data = collector.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=start_time,
                        end_time=end_time,
                        limit=1000
                    )
                    
                    if data is not None and len(data) > 100:
                        # Use close prices and clean symbol name
                        clean_symbol = symbol.replace('USDT', '')
                        interval_data[clean_symbol] = data['close']
                        
                except Exception as e:
                    print(f"    Failed to collect {symbol} {interval}: {e}")
                    continue
            
            if len(interval_data) >= 2:  # Need at least 2 assets
                # Align data
                df = pd.DataFrame(interval_data)
                df = df.dropna()
                
                if len(df) > 100:
                    collected_data[interval] = df
                    print(f"    Successfully collected {interval}: {df.shape}")
                else:
                    print(f"    Insufficient data for {interval}: {len(df)} rows")
        
        return collected_data
        
    except Exception as e:
        print(f"Data collection failed: {e}")
        print("Using synthetic data instead...")
        return generate_synthetic_multi_timeframe_data()


def generate_synthetic_multi_timeframe_data():
    """Generate synthetic data for multiple timeframes"""
    print("Generating synthetic multi-timeframe data...")
    
    np.random.seed(42)
    base_periods = 2000  # Base number of observations
    
    # Generate base high-frequency data
    dates_15m = pd.date_range(start='2023-01-01', periods=base_periods, freq='15min')
    
    # Create correlated crypto-like price series
    returns_btc = np.random.normal(0, 0.02, base_periods)
    price_btc = 40000 * np.exp(np.cumsum(returns_btc))
    
    # ETH with some cointegration relationship to BTC
    beta_eth = 0.7 + 0.1 * np.sin(np.arange(base_periods) / 100)  # Time-varying relationship
    returns_eth = 0.8 * returns_btc + np.random.normal(0, 0.015, base_periods)
    price_eth = 2500 * np.exp(np.cumsum(returns_eth))
    
    # ADA with weaker relationship
    returns_ada = 0.3 * returns_btc + np.random.normal(0, 0.03, base_periods)
    price_ada = 0.5 * np.exp(np.cumsum(returns_ada))
    
    base_data = pd.DataFrame({
        'BTC': price_btc,
        'ETH': price_eth,
        'ADA': returns_ada
    }, index=dates_15m)
    
    # Create different timeframe data by resampling
    timeframes = {
        '15m': base_data,
        '1h': base_data.resample('1H').last().dropna(),
        '4h': base_data.resample('4H').last().dropna(),
        '1d': base_data.resample('1D').last().dropna()
    }
    
    for tf, data in timeframes.items():
        print(f"  {tf}: {data.shape} observations")
        
    return timeframes


def test_rolling_cointegration(data: pd.DataFrame, 
                             lookback_window: int = 252,
                             step_size: int = 30) -> Dict:
    """Test cointegration stability over rolling windows"""
    print(f"Testing rolling cointegration with {lookback_window}d window...")
    
    results = []
    total_windows = (len(data) - lookback_window) // step_size
    
    for i in range(0, len(data) - lookback_window, step_size):
        window_data = data.iloc[i:i + lookback_window]
        window_end = data.index[i + lookback_window - 1]
        
        try:
            analyzer = CointegrationAnalyzer(
                price_data=window_data,
                confidence_level=0.10  # More lenient for rolling tests
            )
            
            # Test pairwise cointegration
            pairs_results = analyzer.analyze_pairs_cointegration()
            
            # Extract key metrics for each pair
            window_result = {
                'window_end': window_end,
                'window_start': data.index[i],
                'total_pairs': len(pairs_results),
                'cointegrated_pairs': sum(1 for p in pairs_results.values() if p['is_cointegrated']),
                'pairs_detail': {}
            }
            
            for pair_name, pair_info in pairs_results.items():
                window_result['pairs_detail'][pair_name] = {
                    'is_cointegrated': pair_info['is_cointegrated'],
                    'p_value': pair_info['min_p_value'],
                    'coef': pair_info[f'test_{pair_info["best_direction"]}']['cointegration_coefficient']
                }
            
            results.append(window_result)
            
            # Progress indicator
            if len(results) % 5 == 0:
                print(f"    Progress: {len(results)}/{total_windows} windows")
                
        except Exception as e:
            print(f"    Window {i} failed: {e}")
            continue
    
    return {
        'rolling_results': results,
        'lookback_window': lookback_window,
        'step_size': step_size
    }


def comprehensive_parameter_testing(data: pd.DataFrame) -> pd.DataFrame:
    """Test multiple parameter combinations comprehensively"""
    print(f"Running comprehensive parameter testing on data: {data.shape}")
    
    # Parameter combinations to test
    test_params = {
        'lookback_periods': [60, 120, 200, 300],  # Different lookback windows
        'zscore_windows': [30, 60, 90, 120],      # Different z-score calculation windows
        'entry_thresholds': [1.5, 2.0, 2.5],     # Entry thresholds
        'exit_thresholds': [0.3, 0.5, 0.8],      # Exit thresholds
        'confidence_levels': [0.05, 0.10]        # Statistical significance levels
    }
    
    # Generate all combinations
    param_combinations = list(itertools.product(
        test_params['lookback_periods'],
        test_params['zscore_windows'], 
        test_params['entry_thresholds'],
        test_params['exit_thresholds'],
        test_params['confidence_levels']
    ))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    results = []
    
    for i, (lookback, zscore_win, entry_thresh, exit_thresh, conf_level) in enumerate(param_combinations):
        try:
            # Initialize analyzer with current parameters
            analyzer = CointegrationAnalyzer(
                price_data=data,
                lookback_period=lookback,
                confidence_level=conf_level
            )
            
            # Test cointegration
            pairs_results = analyzer.analyze_pairs_cointegration()
            
            cointegrated_pairs = [(k, v) for k, v in pairs_results.items() if v['is_cointegrated']]
            
            # Calculate metrics for this parameter set
            param_result = {
                'lookback_period': lookback,
                'zscore_window': zscore_win,
                'entry_threshold': entry_thresh,
                'exit_threshold': exit_thresh,
                'confidence_level': conf_level,
                'total_pairs_tested': len(pairs_results),
                'cointegrated_pairs_count': len(cointegrated_pairs),
                'cointegration_rate': len(cointegrated_pairs) / len(pairs_results) if pairs_results else 0,
                'avg_p_value': np.mean([p['min_p_value'] for p in pairs_results.values()]),
                'min_p_value': min([p['min_p_value'] for p in pairs_results.values()]) if pairs_results else 1.0
            }
            
            # Test signal generation on best pair (if exists)
            if cointegrated_pairs:
                best_pair_key = cointegrated_pairs[0][0]  # Take first cointegrated pair
                
                try:
                    # Test both fixed and dynamic thresholds
                    signals_fixed = analyzer.generate_trading_signals(
                        pair_key=best_pair_key,
                        entry_threshold=entry_thresh,
                        exit_threshold=exit_thresh,
                        lookback_zscore=zscore_win,
                        dynamic_threshold=False
                    )
                    
                    signals_dynamic = analyzer.generate_trading_signals(
                        pair_key=best_pair_key,
                        lookback_zscore=zscore_win,
                        dynamic_threshold=True
                    )
                    
                    # Signal quality metrics
                    param_result.update({
                        'fixed_trades': (signals_fixed['entry_signal'] != 0).sum(),
                        'dynamic_trades': (signals_dynamic['entry_signal'] != 0).sum(),
                        'fixed_valid_zscores': signals_fixed['zscore'].notna().sum(),
                        'dynamic_valid_zscores': signals_dynamic['zscore'].notna().sum(),
                        'stop_loss_signals': (signals_dynamic['stop_loss_signal'] != 0).sum() if 'stop_loss_signal' in signals_dynamic else 0
                    })
                    
                    # Calculate basic performance
                    if param_result['fixed_trades'] > 0:
                        perf = analyzer.calculate_trading_performance(
                            signals=signals_fixed,
                            pair_key=best_pair_key
                        )
                        param_result.update({
                            'fixed_total_return': perf['total_return'],
                            'fixed_sharpe_ratio': perf['sharpe_ratio'],
                            'fixed_max_drawdown': perf['max_drawdown'],
                            'fixed_hit_rate': perf['hit_rate']
                        })
                    
                except Exception as e:
                    print(f"    Signal generation failed for combo {i+1}: {e}")
                    param_result.update({
                        'fixed_trades': 0,
                        'dynamic_trades': 0,
                        'fixed_valid_zscores': 0,
                        'dynamic_valid_zscores': 0
                    })
            
            results.append(param_result)
            
            # Progress updates
            if (i + 1) % 20 == 0:
                print(f"    Completed {i+1}/{len(param_combinations)} combinations")
                
        except Exception as e:
            print(f"    Parameter combination {i+1} failed: {e}")
            continue
    
    return pd.DataFrame(results)


def analyze_timeframe_effects(multi_timeframe_data: Dict) -> Dict:
    """Analyze the effect of different timeframes on cointegration"""
    print("Analyzing timeframe effects on cointegration...")
    
    timeframe_results = {}
    
    for timeframe, data in multi_timeframe_data.items():
        print(f"\nAnalyzing {timeframe} timeframe ({data.shape})...")
        
        try:
            # Basic cointegration analysis
            analyzer = CointegrationAnalyzer(price_data=data)
            pairs_results = analyzer.analyze_pairs_cointegration()
            
            # Parameter sensitivity analysis
            param_results = comprehensive_parameter_testing(data)
            
            # Rolling stability test (if enough data)
            rolling_results = None
            if len(data) > 300:
                rolling_results = test_rolling_cointegration(data, lookback_window=min(252, len(data)//2))
            
            timeframe_results[timeframe] = {
                'data_shape': data.shape,
                'basic_results': pairs_results,
                'parameter_sensitivity': param_results,
                'rolling_stability': rolling_results,
                'summary': {
                    'cointegrated_pairs': sum(1 for p in pairs_results.values() if p['is_cointegrated']),
                    'total_pairs': len(pairs_results),
                    'best_p_value': min([p['min_p_value'] for p in pairs_results.values()]) if pairs_results else 1.0,
                    'avg_r_squared': np.mean([p[f'test_{p["best_direction"]}']['r_squared'] 
                                           for p in pairs_results.values()])
                }
            }
            
            print(f"  {timeframe} Summary:")
            print(f"    Cointegrated pairs: {timeframe_results[timeframe]['summary']['cointegrated_pairs']}")
            print(f"    Best p-value: {timeframe_results[timeframe]['summary']['best_p_value']:.6f}")
            print(f"    Avg R-squared: {timeframe_results[timeframe]['summary']['avg_r_squared']:.4f}")
            
        except Exception as e:
            print(f"  Failed to analyze {timeframe}: {e}")
            continue
    
    return timeframe_results


def create_comprehensive_report(timeframe_results: Dict):
    """Create comprehensive analysis report and visualizations"""
    print("\nCreating comprehensive analysis report...")
    
    # Create summary report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MULTI-TIMEFRAME COINTEGRATION ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Timeframe comparison summary
    report_lines.append("TIMEFRAME COMPARISON SUMMARY")
    report_lines.append("-" * 40)
    
    for timeframe, results in timeframe_results.items():
        if 'summary' in results:
            summary = results['summary']
            report_lines.append(f"\n{timeframe.upper()} Timeframe:")
            report_lines.append(f"  Data Shape: {results['data_shape']}")
            report_lines.append(f"  Cointegrated Pairs: {summary['cointegrated_pairs']}/{summary['total_pairs']}")
            report_lines.append(f"  Success Rate: {summary['cointegrated_pairs']/summary['total_pairs']*100:.1f}%")
            report_lines.append(f"  Best p-value: {summary['best_p_value']:.6f}")
            report_lines.append(f"  Average R²: {summary['avg_r_squared']:.4f}")
    
    # Parameter sensitivity insights
    report_lines.append("\n\nPARAMETER SENSITIVITY INSIGHTS")
    report_lines.append("-" * 40)
    
    for timeframe, results in timeframe_results.items():
        if 'parameter_sensitivity' in results and results['parameter_sensitivity'] is not None:
            param_df = results['parameter_sensitivity']
            if not param_df.empty:
                report_lines.append(f"\n{timeframe.upper()} Parameter Optimization:")
                
                # Best configuration by cointegration detection
                best_cointegration = param_df.loc[param_df['cointegration_rate'].idxmax()]
                report_lines.append(f"  Best for Detection:")
                report_lines.append(f"    Lookback: {best_cointegration['lookback_period']} periods")
                report_lines.append(f"    Z-score Window: {best_cointegration['zscore_window']}")
                report_lines.append(f"    Confidence: {best_cointegration['confidence_level']}")
                report_lines.append(f"    Detection Rate: {best_cointegration['cointegration_rate']:.2%}")
                
                # Best configuration by trading performance (if available)
                if 'fixed_sharpe_ratio' in param_df.columns:
                    valid_performance = param_df.dropna(subset=['fixed_sharpe_ratio'])
                    if not valid_performance.empty:
                        best_performance = valid_performance.loc[valid_performance['fixed_sharpe_ratio'].idxmax()]
                        report_lines.append(f"  Best for Trading:")
                        report_lines.append(f"    Entry Threshold: {best_performance['entry_threshold']}")
                        report_lines.append(f"    Exit Threshold: {best_performance['exit_threshold']}")
                        report_lines.append(f"    Sharpe Ratio: {best_performance['fixed_sharpe_ratio']:.3f}")
    
    # Save report
    report_text = "\n".join(report_lines)
    
    with open('results/model_outputs/multi_timeframe_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Create visualizations
    create_analysis_visualizations(timeframe_results)
    
    print("  Report saved: results/model_outputs/multi_timeframe_analysis_report.txt")
    print("  Visualizations saved to: results/plots/")
    
    return report_text


def create_analysis_visualizations(timeframe_results: Dict):
    """Create comprehensive visualizations"""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Timeframe Cointegration Analysis', fontsize=16)
        
        # Plot 1: Cointegration detection rates by timeframe
        timeframes = []
        success_rates = []
        total_pairs = []
        
        for tf, results in timeframe_results.items():
            if 'summary' in results:
                timeframes.append(tf)
                success_rates.append(results['summary']['cointegrated_pairs']/results['summary']['total_pairs']*100)
                total_pairs.append(results['summary']['total_pairs'])
        
        if timeframes:
            axes[0, 0].bar(timeframes, success_rates, alpha=0.7)
            axes[0, 0].set_title('Cointegration Detection Rate by Timeframe')
            axes[0, 0].set_ylabel('Success Rate (%)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add data points count as text
            for i, (tf, rate, total) in enumerate(zip(timeframes, success_rates, total_pairs)):
                axes[0, 0].text(i, rate + 1, f'{total} pairs', ha='center', fontsize=9)
        
        # Plot 2: Best p-values comparison
        if timeframes:
            p_values = [timeframe_results[tf]['summary']['best_p_value'] for tf in timeframes]
            axes[0, 1].bar(timeframes, p_values, alpha=0.7, color='orange')
            axes[0, 1].set_title('Best p-values by Timeframe')
            axes[0, 1].set_ylabel('p-value')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Parameter sensitivity heatmap (if data available)
        param_data_available = False
        for tf, results in timeframe_results.items():
            if 'parameter_sensitivity' in results and results['parameter_sensitivity'] is not None:
                param_df = results['parameter_sensitivity']
                if not param_df.empty:
                    # Create heatmap of cointegration rate vs parameters
                    pivot = param_df.pivot_table(
                        values='cointegration_rate', 
                        index='lookback_period', 
                        columns='confidence_level', 
                        aggfunc='mean'
                    )
                    
                    if not pivot.empty:
                        im = axes[1, 0].imshow(pivot.values, cmap='viridis', aspect='auto')
                        axes[1, 0].set_title(f'Parameter Sensitivity ({tf})')
                        axes[1, 0].set_xlabel('Confidence Level')
                        axes[1, 0].set_ylabel('Lookback Period')
                        axes[1, 0].set_xticks(range(len(pivot.columns)))
                        axes[1, 0].set_xticklabels(pivot.columns)
                        axes[1, 0].set_yticks(range(len(pivot.index)))
                        axes[1, 0].set_yticklabels(pivot.index)
                        
                        # Add colorbar
                        plt.colorbar(im, ax=axes[1, 0], label='Detection Rate')
                        param_data_available = True
                        break
        
        if not param_data_available:
            axes[1, 0].text(0.5, 0.5, 'Parameter sensitivity\ndata not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Parameter Sensitivity')
        
        # Plot 4: Rolling stability (if available)
        rolling_data_available = False
        for tf, results in timeframe_results.items():
            if 'rolling_stability' in results and results['rolling_stability'] is not None:
                rolling_results = results['rolling_stability']['rolling_results']
                if rolling_results:
                    dates = [r['window_end'] for r in rolling_results]
                    coint_counts = [r['cointegrated_pairs'] for r in rolling_results]
                    
                    axes[1, 1].plot(dates, coint_counts, marker='o', alpha=0.7)
                    axes[1, 1].set_title(f'Rolling Cointegration Stability ({tf})')
                    axes[1, 1].set_xlabel('Date')
                    axes[1, 1].set_ylabel('Cointegrated Pairs Count')
                    axes[1, 1].grid(True, alpha=0.3)
                    rolling_data_available = True
                    break
        
        if not rolling_data_available:
            axes[1, 1].text(0.5, 0.5, 'Rolling stability\ndata not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Rolling Stability')
        
        plt.tight_layout()
        plt.savefig('results/plots/multi_timeframe_cointegration_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Visualization creation failed: {e}")


def main():
    """Main function to run comprehensive multi-timeframe testing"""
    print("=== Multi-Timeframe Cointegration Testing ===")
    
    # Step 1: Collect or generate multi-timeframe data
    print("\n1. Data Collection/Generation...")
    multi_timeframe_data = collect_multi_timeframe_data()
    
    if not multi_timeframe_data:
        print("No data available for analysis.")
        return
    
    print(f"Available timeframes: {list(multi_timeframe_data.keys())}")
    
    # Step 2: Comprehensive analysis
    print("\n2. Running comprehensive multi-timeframe analysis...")
    timeframe_results = analyze_timeframe_effects(multi_timeframe_data)
    
    # Step 3: Generate report and visualizations
    print("\n3. Generating comprehensive report...")
    report = create_comprehensive_report(timeframe_results)
    
    print("\n" + "="*60)
    print("MULTI-TIMEFRAME ANALYSIS COMPLETE")
    print("="*60)
    print("Check results/model_outputs/ for detailed reports")
    print("Check results/plots/ for visualizations")
    print("="*60)
    
    return timeframe_results


if __name__ == "__main__":
    results = main()
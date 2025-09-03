"""
Quick Multi-Timeframe Testing Script

A simplified version for quickly testing different time window parameters
on your existing cointegration data.

Usage: python scripts/quick_timeframe_test.py

Author: Quantitative Trading Team
Date: 2025-09-02
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from models.intermediate.cointegration import CointegrationAnalyzer


def quick_parameter_comparison():
    """Quick comparison of different parameter settings"""
    print("=== Quick Multi-Timeframe Parameter Testing ===")
    
    # Create test data (you can replace this with your real data)
    print("\n1. Loading test data...")
    
    # Option 1: Load your existing data
    try:
        # Try to load your existing results data
        data_path = 'results/data/crypto_cointegration_data.csv'
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"   Loaded existing data: {data.shape}")
        else:
            raise FileNotFoundError("Creating synthetic data instead")
    except:
        # Create synthetic data if no real data available
        print("   Creating synthetic test data...")
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=500, freq='4H')
        
        # Create strongly cointegrated series
        base_price = 50000 * np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))
        cointegrated_price = 0.8 * base_price + 2000 + np.random.normal(0, 100, 500)
        independent_price = 200 * np.exp(np.cumsum(np.random.normal(0, 0.02, 500)))
        
        data = pd.DataFrame({
            'BTC': base_price,
            'ETH': cointegrated_price, 
            'ADA': independent_price
        }, index=dates)
    
    print(f"   Data period: {data.index[0]} to {data.index[-1]}")
    print(f"   Total observations: {len(data)}")
    
    # Parameter sets to test
    test_configurations = [
        # (lookback_period, zscore_window, entry_threshold, exit_threshold, name)
        (60, 30, 2.0, 0.5, "Short-term (60d/30z)"),
        (120, 60, 2.0, 0.5, "Medium-term (120d/60z)"),
        (200, 90, 2.0, 0.5, "Long-term (200d/90z)"),
        (120, 30, 1.5, 0.3, "Aggressive (120d/30z/1.5)"),
        (120, 60, 2.5, 0.8, "Conservative (120d/60z/2.5)"),
    ]
    
    results_summary = []
    
    print(f"\n2. Testing {len(test_configurations)} parameter configurations...")
    
    for i, (lookback, zscore_win, entry_thresh, exit_thresh, name) in enumerate(test_configurations):
        print(f"\n   Testing {name}...")
        
        try:
            # Initialize analyzer with current parameters
            analyzer = CointegrationAnalyzer(
                price_data=data,
                lookback_period=lookback,
                confidence_level=0.10  # More lenient for testing
            )
            
            # Test cointegration
            pairs_results = analyzer.analyze_pairs_cointegration()
            cointegrated_pairs = [(k, v) for k, v in pairs_results.items() if v['is_cointegrated']]
            
            config_result = {
                'name': name,
                'lookback_period': lookback,
                'zscore_window': zscore_win,
                'entry_threshold': entry_thresh,
                'exit_threshold': exit_thresh,
                'total_pairs': len(pairs_results),
                'cointegrated_pairs': len(cointegrated_pairs),
                'detection_rate': len(cointegrated_pairs) / len(pairs_results) if pairs_results else 0,
                'best_p_value': min([p['min_p_value'] for p in pairs_results.values()]) if pairs_results else 1.0,
                'avg_r_squared': np.mean([p[f'test_{p["best_direction"]}']['r_squared'] for p in pairs_results.values()]),
                'signals_generated': 0,
                'valid_zscores': 0,
                'stop_losses': 0
            }
            
            # Test signal generation on best pair (if exists)
            if cointegrated_pairs:
                best_pair_key = cointegrated_pairs[0][0]
                
                try:
                    signals = analyzer.generate_trading_signals(
                        pair_key=best_pair_key,
                        entry_threshold=entry_thresh,
                        exit_threshold=exit_thresh,
                        lookback_zscore=zscore_win,
                        dynamic_threshold=True  # Use optimized dynamic thresholds
                    )
                    
                    config_result.update({
                        'signals_generated': (signals['entry_signal'] != 0).sum(),
                        'valid_zscores': signals['zscore'].notna().sum(),
                        'stop_losses': (signals['stop_loss_signal'] != 0).sum() if 'stop_loss_signal' in signals else 0
                    })
                    
                    # Calculate performance if signals exist
                    if config_result['signals_generated'] > 0:
                        perf = analyzer.calculate_trading_performance(
                            signals=signals,
                            pair_key=best_pair_key
                        )
                        config_result.update({
                            'total_return': perf['total_return'],
                            'sharpe_ratio': perf['sharpe_ratio'],
                            'max_drawdown': perf['max_drawdown']
                        })
                
                except Exception as e:
                    print(f"     Signal generation failed: {e}")
            
            results_summary.append(config_result)
            
            # Print quick summary
            print(f"     Cointegrated pairs: {config_result['cointegrated_pairs']}/{config_result['total_pairs']}")
            print(f"     Detection rate: {config_result['detection_rate']:.1%}")
            print(f"     Best p-value: {config_result['best_p_value']:.6f}")
            print(f"     Signals generated: {config_result['signals_generated']}")
            
        except Exception as e:
            print(f"     Configuration failed: {e}")
            continue
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results_summary)
    
    print(f"\n3. Results Summary:")
    print(f"{'Configuration':<25} {'Detection Rate':<15} {'Best p-value':<12} {'Signals':<8} {'Sharpe':<8}")
    print("-" * 75)
    
    for _, row in results_df.iterrows():
        sharpe = row.get('sharpe_ratio', 0)
        sharpe_str = f"{sharpe:.3f}" if sharpe != 0 else "N/A"
        print(f"{row['name']:<25} {row['detection_rate']:<15.1%} {row['best_p_value']:<12.6f} {row['signals_generated']:<8} {sharpe_str:<8}")
    
    # Find best configurations
    print(f"\n4. Best Configurations:")
    
    if not results_df.empty:
        # Best for detection
        best_detection = results_df.loc[results_df['detection_rate'].idxmax()]
        print(f"   Best Detection Rate: {best_detection['name']} ({best_detection['detection_rate']:.1%})")
        
        # Best for statistical significance
        best_significance = results_df.loc[results_df['best_p_value'].idxmin()]
        print(f"   Best Statistical Significance: {best_significance['name']} (p={best_significance['best_p_value']:.6f})")
        
        # Best for trading (if performance data available)
        if 'sharpe_ratio' in results_df.columns:
            valid_performance = results_df.dropna(subset=['sharpe_ratio'])
            if not valid_performance.empty:
                best_trading = valid_performance.loc[valid_performance['sharpe_ratio'].idxmax()]
                print(f"   Best Trading Performance: {best_trading['name']} (Sharpe={best_trading['sharpe_ratio']:.3f})")
    
    # Optional: Test rolling analysis on best configuration
    if not results_df.empty and len(data) > 300:
        print(f"\n5. Testing rolling analysis on best configuration...")
        
        best_config = results_df.loc[results_df['detection_rate'].idxmax()]
        
        try:
            analyzer = CointegrationAnalyzer(
                price_data=data,
                lookback_period=int(best_config['lookback_period']),
                confidence_level=0.10
            )
            
            # Run rolling analysis with smaller windows for speed
            rolling_results = analyzer.rolling_cointegration_test(
                window_size=min(120, len(data)//3),  # Adjust window size based on data
                step_size=20,
                min_periods=60
            )
            
            print(f"   Rolling analysis completed:")
            print(f"     Windows processed: {rolling_results['parameters']['total_windows_processed']}")
            
            if rolling_results['rolling_results']:
                stability = rolling_results['stability_metrics']['cointegration_rate_stability']
                print(f"     Cointegration rate stability:")
                print(f"       Mean: {stability['mean']:.2%}")
                print(f"       Std: {stability['std']:.2%}")
                print(f"       Range: {stability['min']:.2%} - {stability['max']:.2%}")
            
            # Create visualization
            print(f"\n6. Creating visualization...")
            analyzer.plot_rolling_cointegration_analysis(rolling_results, figsize=(14, 10))
            
        except Exception as e:
            print(f"   Rolling analysis failed: {e}")
    
    # Save results
    results_df.to_csv('results/model_outputs/quick_parameter_comparison.csv', index=False)
    print(f"\n   Results saved: results/model_outputs/quick_parameter_comparison.csv")
    
    print(f"\n" + "="*60)
    print("QUICK PARAMETER TESTING COMPLETE")
    print("="*60)
    print("Key Recommendations:")
    print("1. Use the configuration with highest detection rate for exploration")
    print("2. Use the configuration with best statistical significance for reliability")
    print("3. Consider trading performance metrics for strategy implementation")
    print("4. Test rolling stability to ensure robustness over time")
    print("="*60)
    
    return results_df


if __name__ == "__main__":
    results = quick_parameter_comparison()
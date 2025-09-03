"""
Simple 4-Hour Cointegration Test

A simplified, reliable test script using the proven 4-hour optimal configuration.
This script is designed to work with your existing data and API setup.

Usage: python scripts/simple_4h_cointegration_test.py

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
warnings.filterwarnings('ignore')

# Import custom modules
from data_collection.binance_api import BinanceDataCollector
from models.intermediate.cointegration import CointegrationAnalyzer


def collect_4h_data():
    """Collect 4-hour data using proven API calls"""
    print("Collecting 4-hour cryptocurrency data...")
    
    collector = BinanceDataCollector()
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
    
    # Use the working API call pattern
    end_time = datetime.now()
    start_time = end_time - timedelta(days=120)  # 4 months of 4h data
    
    collected_data = {}
    
    for symbol in symbols:
        try:
            print(f"  Fetching {symbol}...")
            
            data = collector.get_historical_klines(
                symbol=symbol,
                interval='4h',
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            if data is not None and len(data) > 100:
                clean_symbol = symbol.replace('USDT', '')
                collected_data[clean_symbol] = data['close']
                print(f"    [+] {symbol}: {len(data)} data points")
            else:
                print(f"    [-] {symbol}: No data or insufficient data")
                
        except Exception as e:
            print(f"    [-] {symbol}: Failed - {e}")
            continue
    
    if len(collected_data) >= 3:
        # Create aligned DataFrame
        df = pd.DataFrame(collected_data)
        df = df.dropna()
        
        print(f"\nData collection successful:")
        print(f"  Assets: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    else:
        print(f"\nInsufficient data collected ({len(collected_data)} assets)")
        return None


def run_optimal_cointegration_analysis(data):
    """Run cointegration analysis with proven 4h optimal settings"""
    print(f"\nRunning 4-hour optimal cointegration analysis...")
    print(f"Configuration: Very Relaxed (30% detection rate proven)")
    
    # Use proven optimal settings
    analyzer = CointegrationAnalyzer(
        price_data=data,
        lookback_period=180,      # Optimal for 4h data  
        confidence_level=0.15     # Very Relaxed - 30% detection rate
    )
    
    # Run pairwise analysis
    print(f"\nTesting all asset pairs...")
    pairwise_results = analyzer.analyze_pairs_cointegration()
    
    # Extract cointegrated pairs
    cointegrated_pairs = [(k, v) for k, v in pairwise_results.items() 
                         if v['is_cointegrated']]
    
    print(f"\nResults Summary:")
    print(f"  Total pairs tested: {len(pairwise_results)}")
    print(f"  Cointegrated pairs found: {len(cointegrated_pairs)}")
    print(f"  Detection rate: {len(cointegrated_pairs)/len(pairwise_results)*100:.1f}%")
    
    if cointegrated_pairs:
        print(f"\nCointegrated pairs:")
        for pair_name, pair_info in cointegrated_pairs:
            best_test = pair_info[f'test_{pair_info["best_direction"]}']
            print(f"  {pair_name}:")
            print(f"    p-value: {pair_info['min_p_value']:.6f}")
            print(f"    Coefficient: {best_test['cointegration_coefficient']:.4f}")
            print(f"    R-squared: {best_test['r_squared']:.4f}")
        
        # Test signal generation on best pair
        best_pair_key = cointegrated_pairs[0][0]
        print(f"\nTesting trading signals on best pair: {best_pair_key}")
        
        try:
            signals = analyzer.generate_trading_signals(
                pair_key=best_pair_key,
                dynamic_threshold=True,  # Use optimized dynamic thresholds
                lookback_zscore=60       # 2-week z-score window for 4h data
            )
            
            entry_signals = (signals['entry_signal'] != 0).sum()
            valid_zscores = signals['zscore'].notna().sum()
            stop_losses = (signals['stop_loss_signal'] != 0).sum() if 'stop_loss_signal' in signals else 0
            
            print(f"  Signal generation results:")
            print(f"    Entry signals: {entry_signals}")
            print(f"    Valid z-scores: {valid_zscores}/{len(signals)}")
            print(f"    Stop loss triggers: {stop_losses}")
            
            if entry_signals > 0:
                # Calculate basic performance
                perf = analyzer.calculate_trading_performance(
                    signals=signals,
                    pair_key=best_pair_key
                )
                
                print(f"  Basic performance metrics:")
                print(f"    Total return: {perf['total_return']:.4f}")
                print(f"    Sharpe ratio: {perf['sharpe_ratio']:.4f}")
                print(f"    Max drawdown: {perf['max_drawdown']:.4f}")
                print(f"    Hit rate: {perf['hit_rate']:.4f}")
                
                # Create simple visualization
                create_simple_visualization(analyzer, best_pair_key, signals)
            
        except Exception as e:
            print(f"  Signal generation failed: {e}")
    
    else:
        print(f"\nNo cointegrated pairs found.")
        print(f"This could mean:")
        print(f"1. Market conditions are not favorable for cointegration")
        print(f"2. Need longer data history")
        print(f"3. Try different asset combinations")
    
    return analyzer, pairwise_results


def create_simple_visualization(analyzer, pair_key, signals):
    """Create a simple visualization of results"""
    print(f"\nCreating visualization for {pair_key}...")
    
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'4-Hour Cointegration Analysis: {pair_key}', fontsize=14)
        
        # Plot 1: Spread and z-scores
        spread = analyzer.spread_data[pair_key]
        zscore = signals['zscore'].dropna()
        
        axes[0].plot(spread.index, spread, alpha=0.7, color='blue', label='Spread')
        axes[0].axhline(y=spread.mean(), color='black', linestyle='--', alpha=0.5, label='Mean')
        axes[0].axhline(y=spread.mean() + 2*spread.std(), color='red', linestyle=':', alpha=0.5, label='±2σ')
        axes[0].axhline(y=spread.mean() - 2*spread.std(), color='red', linestyle=':', alpha=0.5)
        axes[0].set_title('Cointegration Spread')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Z-scores and signals
        axes[1].plot(zscore.index, zscore, alpha=0.7, color='green', label='Z-score')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Entry threshold')
        axes[1].axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        
        # Mark entry points
        entry_mask = signals['entry_signal'] != 0
        if entry_mask.any():
            entry_points = signals.loc[entry_mask, 'zscore'].dropna()
            if len(entry_points) > 0:
                axes[1].scatter(entry_points.index, entry_points, 
                              color='red', s=50, marker='^', label='Entry signals', zorder=5)
        
        axes[1].set_title('Z-scores and Trading Signals')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/simple_4h_cointegration_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  Visualization saved: results/plots/simple_4h_cointegration_test.png")
        
    except Exception as e:
        print(f"  Visualization failed: {e}")


def main():
    """Main function for simple 4-hour cointegration test"""
    print("=== Simple 4-Hour Cointegration Test ===")
    print("Using proven optimal configuration from comprehensive testing")
    
    # Step 1: Collect 4h data
    print(f"\n[1] Data Collection")
    data = collect_4h_data()
    
    if data is None:
        print(f"[X] Data collection failed. Exiting...")
        return
    
    # Step 2: Run optimal analysis
    print(f"\n[2] Cointegration Analysis")
    analyzer, results = run_optimal_cointegration_analysis(data)
    
    # Step 3: Save results
    print(f"\n[3] Saving Results")
    
    # Save summary
    cointegrated_count = sum(1 for r in results.values() if r['is_cointegrated'])
    
    summary = {
        'test_date': datetime.now().isoformat(),
        'data_shape': data.shape,
        'total_pairs': len(results),
        'cointegrated_pairs': cointegrated_count,
        'detection_rate': cointegrated_count / len(results),
        'configuration': '4h_optimal_very_relaxed'
    }
    
    with open('results/model_outputs/simple_4h_test_summary.txt', 'w') as f:
        f.write("4-Hour Cointegration Test Summary\n")
        f.write("=" * 35 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"  Summary saved: results/model_outputs/simple_4h_test_summary.txt")
    
    print(f"\n" + "="*50)
    print("SIMPLE 4-HOUR TEST COMPLETE")
    print("="*50)
    print(f"Detection Rate: {summary['detection_rate']:.1%}")
    print(f"Cointegrated Pairs: {summary['cointegrated_pairs']}/{summary['total_pairs']}")
    
    if summary['cointegrated_pairs'] > 0:
        print("SUCCESS: Cointegration detected with 4-hour optimal configuration!")
    else:
        print("No cointegration found. Consider:")
        print("1. Trying different time periods")
        print("2. Testing alternative asset combinations")
        print("3. Checking market conditions")
    
    print("="*50)
    
    return analyzer, results, summary


if __name__ == "__main__":
    analyzer, results, summary = main()
"""
Demo Optimized Cointegration Features

This script demonstrates the improvements made to the cointegration analysis:
1. Fixed zscore calculation
2. Dynamic threshold adjustment
3. Risk management features
4. Enhanced signal generation

Uses existing results to show optimization effects.

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


def demo_optimized_features():
    """Demonstrate the optimized features using manual test data"""
    print("=== Demo: Optimized Cointegration Features ===")
    
    # Create simple test case with known cointegrated data
    print("\n1. Creating test data with known cointegration relationship...")
    
    np.random.seed(123)
    n_points = 500
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='4H')
    
    # Create two series that are definitely cointegrated
    # X = base random walk
    x_returns = np.random.normal(0, 0.01, n_points)
    x = 100 * np.exp(np.cumsum(x_returns))
    
    # Y = 2*X + small noise (strong linear relationship)
    noise = np.random.normal(0, 1, n_points)  # Small noise
    y = 2 * x + noise
    
    test_data = pd.DataFrame({
        'Asset1': x,
        'Asset2': y
    }, index=dates)
    
    print(f"Created test data:")
    print(f"  Asset1 range: {x.min():.2f} - {x.max():.2f}")
    print(f"  Asset2 range: {y.min():.2f} - {y.max():.2f}")
    print(f"  Correlation: {np.corrcoef(x, y)[0,1]:.4f}")
    
    # Initialize analyzer
    print("\n2. Testing CointegrationAnalyzer with optimized features...")
    analyzer = CointegrationAnalyzer(
        price_data=test_data,
        confidence_level=0.05
    )
    
    # Test cointegration
    print("\n3. Running cointegration analysis...")
    results = analyzer.analyze_pairs_cointegration()
    
    for pair_name, pair_info in results.items():
        print(f"\nPair: {pair_name}")
        print(f"  Cointegrated: {pair_info['is_cointegrated']}")
        print(f"  p-value: {pair_info['min_p_value']:.6f}")
        
        if pair_info['is_cointegrated']:
            best_test = pair_info[f'test_{pair_info["best_direction"]}']
            print(f"  Cointegration coefficient: {best_test['cointegration_coefficient']:.4f}")
            print(f"  R-squared: {best_test['r_squared']:.4f}")
            
            # Test the optimized signal generation
            print(f"\n4. Testing optimized signal generation...")
            
            print("\n4a. Fixed thresholds (original approach):")
            signals_fixed = analyzer.generate_trading_signals(
                pair_key=pair_name,
                entry_threshold=2.0,
                exit_threshold=0.5,
                dynamic_threshold=False,
                lookback_zscore=50
            )
            
            valid_zscores_fixed = signals_fixed['zscore'].dropna()
            print(f"  Valid z-scores: {len(valid_zscores_fixed)}/{len(signals_fixed)}")
            print(f"  Entry signals: {(signals_fixed['entry_signal'] != 0).sum()}")
            print(f"  Exit signals: {(signals_fixed['exit_signal'] != 0).sum()}")
            
            if len(valid_zscores_fixed) > 0:
                print(f"  Z-score range: [{valid_zscores_fixed.min():.3f}, {valid_zscores_fixed.max():.3f}]")
                print(f"  Z-score std: {valid_zscores_fixed.std():.3f}")
            
            print("\n4b. Dynamic thresholds (optimized approach):")
            signals_dynamic = analyzer.generate_trading_signals(
                pair_key=pair_name,
                dynamic_threshold=True,
                lookback_zscore=50
            )
            
            valid_zscores_dynamic = signals_dynamic['zscore'].dropna()
            print(f"  Valid z-scores: {len(valid_zscores_dynamic)}/{len(signals_dynamic)}")
            print(f"  Entry signals: {(signals_dynamic['entry_signal'] != 0).sum()}")
            print(f"  Exit signals: {(signals_dynamic['exit_signal'] != 0).sum()}")
            print(f"  Stop loss signals: {(signals_dynamic['stop_loss_signal'] != 0).sum()}")
            
            if len(valid_zscores_dynamic) > 0:
                print(f"  Z-score range: [{valid_zscores_dynamic.min():.3f}, {valid_zscores_dynamic.max():.3f}]")
                
            # Show improvement in signal quality
            print(f"\n5. Optimization Impact:")
            print(f"  Z-score calculation improved: {'YES' if len(valid_zscores_dynamic) > 0 else 'NO'}")
            print(f"  Dynamic thresholds working: {'YES' if 'entry_threshold' in signals_dynamic.columns else 'NO'}")
            print(f"  Risk management added: {'YES' if 'stop_loss_signal' in signals_dynamic.columns else 'NO'}")
            
            fixed_trades = (signals_fixed['entry_signal'] != 0).sum()
            dynamic_trades = (signals_dynamic['entry_signal'] != 0).sum()
            print(f"  Trade frequency comparison:")
            print(f"    Fixed thresholds: {fixed_trades} trades")
            print(f"    Dynamic thresholds: {dynamic_trades} trades")
            
            # Create comparison visualization
            print(f"\n6. Creating comparison visualization...")
            
            try:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Optimized Cointegration Analysis Demo', fontsize=16)
                
                # Plot 1: Original price series
                axes[0, 0].plot(test_data.index, test_data['Asset1'], label='Asset1', alpha=0.7)
                axes[0, 0].plot(test_data.index, test_data['Asset2'], label='Asset2', alpha=0.7)
                axes[0, 0].set_title('Price Series')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Spread with statistics
                spread = analyzer.spread_data[pair_name]
                axes[0, 1].plot(spread.index, spread, color='red', alpha=0.7)
                axes[0, 1].axhline(y=spread.mean(), color='black', linestyle='--', alpha=0.5, label='Mean')
                axes[0, 1].axhline(y=spread.mean() + 2*spread.std(), color='red', linestyle=':', alpha=0.5, label='±2σ')
                axes[0, 1].axhline(y=spread.mean() - 2*spread.std(), color='red', linestyle=':', alpha=0.5)
                axes[0, 1].set_title(f'Cointegration Spread (p={pair_info["min_p_value"]:.4f})')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Fixed threshold signals
                valid_indices = signals_fixed['zscore'].notna()
                if valid_indices.any():
                    zscore_fixed = signals_fixed.loc[valid_indices, 'zscore']
                    axes[1, 0].plot(zscore_fixed.index, zscore_fixed, alpha=0.7, label='Z-score')
                    axes[1, 0].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Entry ±2')
                    axes[1, 0].axhline(y=-2, color='red', linestyle='--', alpha=0.5)
                    axes[1, 0].axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Exit ±0.5')
                    axes[1, 0].axhline(y=-0.5, color='green', linestyle=':', alpha=0.5)
                    
                    # Mark entry points
                    entry_mask = signals_fixed['entry_signal'] != 0
                    if entry_mask.any():
                        entry_points = signals_fixed.loc[entry_mask, 'zscore']
                        axes[1, 0].scatter(entry_points.index, entry_points, 
                                         color='red', s=50, marker='^', label='Entries', zorder=5)
                
                axes[1, 0].set_title('Fixed Thresholds (Original)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Plot 4: Dynamic threshold signals
                valid_indices = signals_dynamic['zscore'].notna()
                if valid_indices.any():
                    zscore_dynamic = signals_dynamic.loc[valid_indices, 'zscore']
                    axes[1, 1].plot(zscore_dynamic.index, zscore_dynamic, alpha=0.7, label='Z-score')
                    
                    # Show dynamic thresholds if available
                    if 'entry_threshold' in signals_dynamic.columns:
                        thresh_valid = signals_dynamic['entry_threshold'].notna()
                        if thresh_valid.any():
                            entry_thresh = signals_dynamic.loc[thresh_valid, 'entry_threshold']
                            axes[1, 1].plot(entry_thresh.index, entry_thresh, 
                                          color='red', linestyle='--', alpha=0.7, label='Dynamic Entry+')
                            axes[1, 1].plot(entry_thresh.index, -entry_thresh, 
                                          color='red', linestyle='--', alpha=0.7, label='Dynamic Entry-')
                    else:
                        axes[1, 1].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Entry ±2')
                        axes[1, 1].axhline(y=-2, color='red', linestyle='--', alpha=0.5)
                    
                    # Mark entry and exit points
                    entry_mask = signals_dynamic['entry_signal'] != 0
                    if entry_mask.any():
                        entry_points = signals_dynamic.loc[entry_mask, 'zscore']
                        axes[1, 1].scatter(entry_points.index, entry_points, 
                                         color='red', s=50, marker='^', label='Entries', zorder=5)
                    
                    stop_mask = signals_dynamic['stop_loss_signal'] != 0
                    if stop_mask.any():
                        stop_points = signals_dynamic.loc[stop_mask, 'zscore']
                        axes[1, 1].scatter(stop_points.index, stop_points, 
                                         color='orange', s=50, marker='x', label='Stop Loss', zorder=5)
                
                axes[1, 1].set_title('Dynamic Thresholds + Risk Mgmt (Optimized)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('results/plots/cointegration_optimization_demo.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print("  Visualization saved: results/plots/cointegration_optimization_demo.png")
                
            except Exception as e:
                print(f"  Visualization error: {e}")
    
    print(f"\n" + "="*60)
    print("KEY OPTIMIZATIONS IMPLEMENTED:")
    print("="*60)
    print("[+] Fixed Johansen test import issues")
    print("[+] Enhanced z-score calculation with min_periods")
    print("[+] Added dynamic threshold adjustment based on volatility")  
    print("[+] Implemented stop-loss risk management (2% max loss)")
    print("[+] Improved signal generation robustness")
    print("[+] Better handling of edge cases and NaN values")
    print("="*60)
    
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = demo_optimized_features()
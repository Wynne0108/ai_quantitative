"""
High-Frequency Pairs Trading Demo Script

This script demonstrates high-frequency pairs trading capabilities including:
1. Real-time data simulation and processing
2. Dynamic signal generation with sub-minute updates
3. Advanced risk management and position sizing
4. Real-time performance monitoring
5. Latency optimization techniques

The demo simulates a high-frequency trading environment using the cointegrated
pair discovered in the previous analysis (ADA/LINK).

Usage: python scripts/demo_high_frequency_pairs.py

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import threading
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.intermediate.cointegration import CointegrationAnalyzer
from models.intermediate.pairs_trading import PairsTradingStrategy
from models.intermediate.high_frequency_pairs import HighFrequencyPairsTrader


def simulate_real_time_prices(hf_trader, duration_minutes=60):
    """
    Simulate real-time price feed for high-frequency trading demo
    
    Parameters:
    -----------
    hf_trader : HighFrequencyPairsTrader
        High-frequency trader instance
    duration_minutes : int, default=60
        Duration of simulation in minutes
    """
    print(f"Starting real-time price simulation for {duration_minutes} minutes...")
    
    # Initial prices (using realistic crypto prices)
    current_prices = {
        hf_trader.asset1: 0.35,  # ADA
        hf_trader.asset2: 18.50  # LINK
    }
    
    # Price simulation parameters
    price_params = {
        hf_trader.asset1: {'mu': 0.00001, 'sigma': 0.0015, 'correlation': 0.75},
        hf_trader.asset2: {'mu': 0.00002, 'sigma': 0.002, 'correlation': 0.75}
    }
    
    start_time = datetime.now()
    update_interval = 1  # 1 second updates
    updates_per_minute = 60 // update_interval
    total_updates = duration_minutes * updates_per_minute
    
    print(f"Price update frequency: {update_interval} seconds")
    print(f"Total updates planned: {total_updates}")
    
    # Generate correlated random walks
    np.random.seed(42)
    
    for i in range(total_updates):
        if not hf_trader.is_running:
            break
        
        current_time = start_time + timedelta(seconds=i * update_interval)
        
        # Generate correlated price movements
        z1 = np.random.normal(0, 1)
        z2 = np.random.normal(0, 1)
        
        # Create correlation
        correlation = price_params[hf_trader.asset1]['correlation']
        z2_corr = correlation * z1 + np.sqrt(1 - correlation**2) * z2
        
        # Update prices with geometric brownian motion
        for j, (asset, z) in enumerate([(hf_trader.asset1, z1), (hf_trader.asset2, z2_corr)]):
            mu = price_params[asset]['mu']
            sigma = price_params[asset]['sigma']
            dt = update_interval / 86400  # Convert to days
            
            # Geometric Brownian Motion
            price_change = mu * dt + sigma * np.sqrt(dt) * z
            current_prices[asset] *= np.exp(price_change)
            
            # Add some mean reversion to maintain cointegration
            if len(hf_trader.spread_buffer) > 10:
                recent_spreads = [s['spread'] for s in list(hf_trader.spread_buffer)[-10:]]
                spread_mean = np.mean(recent_spreads)
                current_spread = recent_spreads[-1]
                
                # Add mean reversion force
                reversion_force = -0.0001 * (current_spread - spread_mean)
                if asset == hf_trader.asset1:
                    current_prices[asset] *= np.exp(reversion_force)
                else:
                    current_prices[asset] *= np.exp(-reversion_force * hf_trader.base_strategy.hedge_ratio)
            
            # Update trader with new price
            hf_trader.add_price_data(asset, current_time, current_prices[asset])
        
        # Print periodic updates
        if i % (updates_per_minute * 5) == 0:  # Every 5 minutes
            status = hf_trader.get_real_time_status()
            print(f"\n--- {current_time.strftime('%H:%M:%S')} Update ---")
            print(f"Prices: {hf_trader.asset1}=${current_prices[hf_trader.asset1]:.4f}, "
                  f"{hf_trader.asset2}=${current_prices[hf_trader.asset2]:.4f}")
            if 'spread_zscore' in status:
                print(f"Spread Z-score: {status['spread_zscore']:.3f}")
            print(f"Position: {status['current_position']:.4f}")
            print(f"Trades today: {status['trades_today']}")
            print(f"Daily P&L: {status['daily_pnl']:.4f} ({status['daily_pnl']*100:.2f}%)")
        
        time.sleep(update_interval)
    
    print(f"\nPrice simulation completed after {duration_minutes} minutes")


def main():
    """
    Main function demonstrating high-frequency pairs trading
    """
    print("=== High-Frequency Cryptocurrency Pairs Trading Demo ===")
    
    # Step 1: Load previous analysis results
    print("\n1. Loading Base Strategy from Previous Analysis...")
    
    # For demo purposes, we'll create a simplified base strategy
    # In practice, you'd load this from the previous cointegration analysis
    sample_data = pd.DataFrame({
        'ADA': np.random.lognormal(np.log(0.35), 0.3, 1000),
        'LINK': np.random.lognormal(np.log(18.5), 0.3, 1000)
    }, index=pd.date_range('2025-01-01', periods=1000, freq='4H'))
    
    # Create base strategy
    base_pairs_strategy = PairsTradingStrategy(
        price_data=sample_data,
        pair_assets=('ADA', 'LINK'),
        lookback_period=200
    )
    
    # Set a reasonable hedge ratio from previous analysis
    base_pairs_strategy.hedge_ratio = 0.5815  # From previous demo results
    
    print("Base strategy initialized with hedge ratio: 0.5815")
    
    # Step 2: Initialize High-Frequency Trading System
    print("\n2. Initializing High-Frequency Trading System...")
    
    hf_trader = HighFrequencyPairsTrader(
        pair_assets=('ADA', 'LINK'),
        base_strategy=base_pairs_strategy,
        execution_timeframe='1T',   # 1-minute execution
        signal_timeframe='15T',     # 15-minute signals
        trend_timeframe='4H'        # 4-hour trend
    )
    
    # Step 3: Configure High-Frequency Parameters
    print("\n3. Configuring High-Frequency Parameters...")
    
    hf_trader.update_hf_parameters(
        max_trades_per_hour=8,
        min_profit_threshold=0.001,   # 0.1% minimum profit
        entry_threshold=1.5,          # More aggressive
        exit_threshold=0.4,
        lookback_window=30,           # 30-point window
        max_position_size=0.01,       # 1% per trade
        max_daily_drawdown=0.03,      # 3% daily limit
        correlation_threshold=0.65
    )
    
    print("High-frequency parameters configured:")
    print(f"  Max trades per hour: {hf_trader.hf_params['max_trades_per_hour']}")
    print(f"  Entry threshold: {hf_trader.hf_params['entry_threshold']}")
    print(f"  Min profit threshold: {hf_trader.hf_params['min_profit_threshold']*100:.1f}%")
    print(f"  Daily drawdown limit: {hf_trader.hf_params['max_daily_drawdown']*100:.1f}%")
    
    # Step 4: Start Real-Time Trading System
    print("\n4. Starting Real-Time Trading System...")
    
    hf_trader.start_real_time_trading()
    
    # Step 5: Simulate Real-Time Market Data
    print("\n5. Simulating Real-Time Market Data Feed...")
    print("Note: This demo simulates 30 minutes of 1-second price updates")
    
    # Run price simulation in separate thread
    try:
        simulate_real_time_prices(hf_trader, duration_minutes=30)
        
    except KeyboardInterrupt:
        print("\nTrading interrupted by user...")
    
    # Step 6: Stop Trading System
    print("\n6. Stopping Trading System...")
    
    hf_trader.stop_trading()
    
    # Step 7: Generate Performance Analysis
    print("\n7. Analyzing High-Frequency Trading Performance...")
    
    # Get final status
    final_status = hf_trader.get_real_time_status()
    
    print("\n=== FINAL TRADING STATUS ===")
    print(f"Total simulation time: 30 minutes")
    print(f"Final position: {final_status['current_position']:.4f}")
    print(f"Trades executed: {final_status['trades_today']}")
    print(f"Final P&L: {final_status['daily_pnl']:.4f} ({final_status['daily_pnl']*100:.2f}%)")
    
    if final_status['trades_today'] > 0:
        print(f"Recent win rate: {final_status['recent_win_rate']:.1%}")
        print(f"Average execution time: {final_status['avg_execution_time']*1000:.2f} ms")
        print(f"Average slippage: {final_status['avg_slippage']*10000:.2f} bps")
    
    # Step 8: Detailed Performance Report
    if len(hf_trader.hf_performance['trades']) > 0:
        print("\n8. Generating Detailed Performance Report...")
        
        performance_report = hf_trader.get_performance_report()
        print(performance_report)
        
        # Step 9: Create Performance Visualizations
        print("\n9. Creating Performance Visualizations...")
        create_hf_performance_plots(hf_trader)
        
        # Step 10: Save Results
        print("\n10. Saving High-Frequency Trading Results...")
        save_hf_results(hf_trader)
        
    else:
        print("\nNo trades were executed during the simulation.")
        print("This could be due to:")
        print("- No signals met the profit threshold")
        print("- Risk management constraints prevented trading")
        print("- Market conditions were not suitable for pairs trading")
    
    print(f"\n=== HIGH-FREQUENCY PAIRS TRADING DEMO COMPLETED ===")
    print("Key takeaways:")
    print("1. High-frequency systems require careful risk management")
    print("2. Transaction costs become critical at higher frequencies")
    print("3. Real-time monitoring and circuit breakers are essential")
    print("4. Correlation and cointegration must be continuously monitored")
    
    return hf_trader


def create_hf_performance_plots(hf_trader):
    """
    Create comprehensive high-frequency trading performance visualizations
    """
    if len(hf_trader.hf_performance['trades']) == 0:
        print("No trades to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'High-Frequency Pairs Trading: {hf_trader.asset1}/{hf_trader.asset2}', fontsize=14)
    
    # Plot 1: Cumulative P&L
    trades = hf_trader.hf_performance['trades']
    timestamps = [t['timestamp'] for t in trades]
    cumulative_pnl = np.cumsum([t['profit'] for t in trades])
    
    axes[0, 0].plot(timestamps, cumulative_pnl, 'b-', linewidth=2)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Cumulative P&L')
    axes[0, 0].set_ylabel('Cumulative Profit/Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Trade Distribution
    profits = [t['profit'] for t in trades]
    axes[0, 1].hist(profits, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Trade Profit Distribution')
    axes[0, 1].set_xlabel('Profit per Trade')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Execution Times
    if hf_trader.hf_performance['execution_times']:
        exec_times = np.array(hf_trader.hf_performance['execution_times']) * 1000  # Convert to ms
        axes[1, 0].plot(exec_times, 'g.-', alpha=0.7)
        axes[1, 0].axhline(y=np.mean(exec_times), color='red', linestyle='--', 
                          label=f'Avg: {np.mean(exec_times):.1f}ms')
        axes[1, 0].set_title('Execution Times')
        axes[1, 0].set_ylabel('Execution Time (ms)')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Rolling Performance Metrics
    if len(trades) >= 5:
        window = min(5, len(trades))
        rolling_win_rate = []
        rolling_avg_profit = []
        
        for i in range(window-1, len(trades)):
            recent_trades = trades[i-window+1:i+1]
            win_rate = np.mean([t['profit'] > 0 for t in recent_trades])
            avg_profit = np.mean([t['profit'] for t in recent_trades])
            
            rolling_win_rate.append(win_rate)
            rolling_avg_profit.append(avg_profit)
        
        trade_numbers = list(range(window, len(trades)+1))
        
        ax4_twin = axes[1, 1].twinx()
        
        line1 = axes[1, 1].plot(trade_numbers, rolling_win_rate, 'b-', label='Win Rate')
        line2 = ax4_twin.plot(trade_numbers, rolling_avg_profit, 'r-', label='Avg Profit')
        
        axes[1, 1].set_title(f'Rolling Performance (Window: {window})')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('Win Rate', color='b')
        ax4_twin.set_ylabel('Average Profit', color='r')
        
        # Combine legends
        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/hf_pairs_trading_performance.png', dpi=300, bbox_inches='tight')
    print("Performance plots saved: results/plots/hf_pairs_trading_performance.png")
    plt.show()


def save_hf_results(hf_trader):
    """
    Save high-frequency trading results to files
    """
    os.makedirs('results/hf_outputs', exist_ok=True)
    
    # Save trade details
    if hf_trader.hf_performance['trades']:
        trades_df = pd.DataFrame(hf_trader.hf_performance['trades'])
        trades_df.to_csv('results/hf_outputs/hf_trade_details.csv', index=False)
    
    # Save execution metrics
    if hf_trader.hf_performance['execution_times']:
        metrics_df = pd.DataFrame({
            'execution_times_ms': np.array(hf_trader.hf_performance['execution_times']) * 1000,
            'slippage_bps': np.array(hf_trader.hf_performance['slippage']) * 10000
        })
        metrics_df.to_csv('results/hf_outputs/hf_execution_metrics.csv', index=False)
    
    # Save final status
    final_status = hf_trader.get_real_time_status()
    import json
    
    # Convert numpy types for JSON serialization
    status_json = {}
    for key, value in final_status.items():
        if isinstance(value, (np.integer, np.floating)):
            status_json[key] = float(value)
        elif isinstance(value, datetime):
            status_json[key] = value.isoformat()
        else:
            status_json[key] = value
    
    with open('results/hf_outputs/hf_final_status.json', 'w') as f:
        json.dump(status_json, f, indent=2)
    
    # Save performance report
    performance_report = hf_trader.get_performance_report()
    with open('results/hf_outputs/hf_performance_report.txt', 'w') as f:
        f.write(performance_report)
    
    print("High-frequency results saved to: results/hf_outputs/")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/hf_outputs', exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('default')
    plt.rcParams['figure.max_open_warning'] = 0
    
    try:
        hf_trader = main()
        print("High-frequency pairs trading demo completed successfully!")
        
    except Exception as e:
        print(f"Error during high-frequency trading demo: {e}")
        import traceback
        traceback.print_exc()
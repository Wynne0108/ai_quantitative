"""
Quick High-Frequency Pairs Trading Demo

This is a simplified version that demonstrates the key concepts without 
real-time execution, suitable for quick testing and understanding.

Usage: python scripts/demo_hf_quick.py

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
import warnings
warnings.filterwarnings('ignore')

from models.intermediate.cointegration import CointegrationAnalyzer
from models.intermediate.pairs_trading import PairsTradingStrategy
from models.intermediate.high_frequency_pairs import HighFrequencyPairsTrader


def generate_hf_test_data():
    """
    Generate high-frequency test data (1-minute intervals)
    """
    print("Generating high-frequency test data (1-minute intervals)...")
    
    # Generate 8 hours of minute data (480 points)
    n_points = 480
    timestamps = pd.date_range('2025-09-01 09:00', periods=n_points, freq='1T')
    
    np.random.seed(42)
    
    # Base prices
    ada_base = 0.35
    link_base = 18.5
    
    # Parameters for correlated GBM
    correlation = 0.8
    dt = 1/1440  # 1 minute in days
    
    # Generate correlated returns
    ada_returns = []
    link_returns = []
    
    for i in range(n_points):
        z1 = np.random.normal(0, 1)
        z2 = np.random.normal(0, 1)
        
        # Create correlation
        z2_corr = correlation * z1 + np.sqrt(1 - correlation**2) * z2
        
        # Add some mean reversion to maintain cointegration
        if i > 20:
            recent_ada = ada_prices[-20:]
            recent_link = link_prices[-20:]
            
            # Simple cointegration spread
            spread = np.log(recent_ada[-1]) - 0.5815 * np.log(recent_link[-1])
            spread_mean = np.mean([np.log(p1) - 0.5815 * np.log(p2) 
                                  for p1, p2 in zip(recent_ada, recent_link)])
            
            # Mean reversion force
            reversion = -0.1 * (spread - spread_mean)
            z1 += reversion * 0.5
            z2_corr -= reversion * 0.5
        
        # Calculate returns
        ada_ret = 0.0001 * dt + 0.02 * np.sqrt(dt) * z1
        link_ret = 0.0002 * dt + 0.025 * np.sqrt(dt) * z2_corr
        
        ada_returns.append(ada_ret)
        link_returns.append(link_ret)
        
        # Calculate prices
        if i == 0:
            ada_prices = [ada_base]
            link_prices = [link_base]
        else:
            ada_prices.append(ada_prices[-1] * np.exp(ada_ret))
            link_prices.append(link_prices[-1] * np.exp(link_ret))
    
    # Create DataFrame
    hf_data = pd.DataFrame({
        'ADA': ada_prices,
        'LINK': link_prices
    }, index=timestamps)
    
    print(f"Generated {len(hf_data)} data points")
    print(f"ADA price range: ${hf_data['ADA'].min():.4f} - ${hf_data['ADA'].max():.4f}")
    print(f"LINK price range: ${hf_data['LINK'].min():.4f} - ${hf_data['LINK'].max():.4f}")
    
    return hf_data


def main():
    """
    Quick demonstration of high-frequency pairs trading concepts
    """
    print("=== Quick High-Frequency Pairs Trading Demo ===")
    
    # Step 1: Generate test data
    print("\n1. Generating High-Frequency Test Data...")
    hf_data = generate_hf_test_data()
    
    # Step 2: Setup base strategy
    print("\n2. Setting Up Base Pairs Trading Strategy...")
    base_strategy = PairsTradingStrategy(
        price_data=hf_data,
        pair_assets=('ADA', 'LINK'),
        lookback_period=100
    )
    
    # Analyze the pair
    pair_analysis = base_strategy.analyze_pair_relationship()
    print(f"Cointegrated: {pair_analysis['is_cointegrated']}")
    
    # For demo purposes, set hedge ratio if not found
    if base_strategy.hedge_ratio is None:
        print("No cointegration found in test data, using fixed hedge ratio for demo")
        base_strategy.hedge_ratio = 0.5815  # Use ratio from previous analysis
    
    print(f"Hedge ratio: {base_strategy.hedge_ratio:.4f}")
    
    # Step 3: Initialize HF trader
    print("\n3. Initializing High-Frequency Trading System...")
    hf_trader = HighFrequencyPairsTrader(
        pair_assets=('ADA', 'LINK'),
        base_strategy=base_strategy,
        execution_timeframe='1T',
        signal_timeframe='15T',
        trend_timeframe='1H'
    )
    
    # Configure for quick demo
    hf_trader.update_hf_parameters(
        entry_threshold=1.0,      # More aggressive
        exit_threshold=0.3,
        min_profit_threshold=0.0005,  # 0.05%
        max_trades_per_hour=20,
        lookback_window=20
    )
    
    # Step 4: Simulate trading on historical data
    print("\n4. Simulating High-Frequency Trading...")
    
    trades_executed = []
    positions = []
    spreads = []
    signals = []
    
    # Feed data point by point
    for i, (timestamp, row) in enumerate(hf_data.iterrows()):
        # Add price data
        hf_trader.add_price_data('ADA', timestamp, row['ADA'])
        hf_trader.add_price_data('LINK', timestamp, row['LINK'])
        
        # Generate signal every few points
        if i > 30 and i % 5 == 0:  # Every 5 minutes
            signal = hf_trader.generate_hf_signals()
            
            if signal:
                signals.append({
                    'timestamp': timestamp,
                    'signal': signal
                })
                
                # Execute trade
                if signal['action'] in ['long_spread', 'short_spread', 'close_position']:
                    execution = hf_trader.execute_trade(signal)
                    
                    if execution['success']:
                        trades_executed.append({
                            'timestamp': timestamp,
                            'action': signal['action'],
                            'position_size': signal.get('position_size', 0),
                            'zscore': signal['zscore'],
                            'confidence': signal['confidence'],
                            'profit': execution.get('profit', 0)
                        })
        
        # Record position and spread
        positions.append(hf_trader.current_position)
        if len(hf_trader.spread_buffer) > 0:
            spreads.append(hf_trader.spread_buffer[-1]['spread'])
        else:
            spreads.append(0)
    
    print(f"Simulation complete! Trades executed: {len(trades_executed)}")
    
    # Step 5: Analyze results
    print("\n5. Analyzing High-Frequency Trading Results...")
    
    if len(trades_executed) > 0:
        trades_df = pd.DataFrame(trades_executed)
        
        # Calculate performance metrics
        total_trades = len(trades_executed)
        profitable_trades = len([t for t in trades_executed if t['profit'] > 0])
        total_profit = sum(t['profit'] for t in trades_executed)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Total Profit: {total_profit:.4f} ({total_profit*100:.2f}%)")
        print(f"Average Profit per Trade: {total_profit/total_trades:.4f}")
        
        # Analyze by trade type
        long_trades = [t for t in trades_executed if t['action'] == 'long_spread']
        short_trades = [t for t in trades_executed if t['action'] == 'short_spread']
        
        print(f"\nTrade Breakdown:")
        print(f"Long Spread Entries: {len(long_trades)}")
        print(f"Short Spread Entries: {len(short_trades)}")
        
        # Calculate execution metrics
        execution_times = hf_trader.hf_performance['execution_times']
        if execution_times:
            avg_exec_time = np.mean(execution_times) * 1000
            print(f"Average Execution Time: {avg_exec_time:.2f} ms")
        
        slippages = hf_trader.hf_performance['slippage']
        if slippages:
            avg_slippage = np.mean(slippages) * 10000
            print(f"Average Slippage: {avg_slippage:.2f} bps")
        
        # Step 6: Create visualizations
        print("\n6. Creating Performance Visualizations...")
        create_quick_hf_plots(hf_data, trades_executed, spreads, positions)
        
    else:
        print("No trades were executed during the simulation.")
        print("This could indicate:")
        print("- Signals didn't meet profit thresholds")
        print("- Market conditions weren't suitable")
        print("- Risk constraints prevented trading")
    
    # Step 7: Compare with base strategy
    print("\n7. Comparing with Base Strategy Performance...")
    
    # Run base strategy on same data (need to set spread_series first)
    if base_strategy.spread_series is None:
        log_prices = base_strategy.cointegration_analyzer.log_prices
        base_strategy.spread_series = log_prices['ADA'] - base_strategy.hedge_ratio * log_prices['LINK']
    
    base_signals = base_strategy.generate_trading_signals(method='zscore')
    base_performance = base_strategy.calculate_performance_metrics()
    
    print(f"\nBase Strategy (4H equivalent) Performance:")
    print(f"Total Return: {base_performance['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {base_performance['sharpe_ratio']:.2f}")
    print(f"Number of Trades: {base_performance['number_of_trades']}")
    
    if len(trades_executed) > 0:
        hf_total_return = sum(t['profit'] for t in trades_executed)
        hf_trades = len(trades_executed)
        
        print(f"\nHigh-Frequency Strategy Performance:")
        print(f"Total Return: {hf_total_return*100:.2f}%")
        print(f"Number of Trades: {hf_trades}")
        print(f"Trade Frequency Increase: {hf_trades/max(1,base_performance['number_of_trades']):.1f}x")
    
    print(f"\n=== KEY INSIGHTS ===")
    print("1. High-frequency trading increases trade opportunities")
    print("2. Transaction costs become more critical at higher frequencies") 
    print("3. Risk management and signal quality are essential")
    print("4. Real-time execution requires careful latency optimization")
    
    return hf_trader, trades_executed


def create_quick_hf_plots(price_data, trades, spreads, positions):
    """
    Create quick visualization of high-frequency trading results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('High-Frequency Pairs Trading Simulation Results', fontsize=14)
    
    # Plot 1: Price series
    axes[0, 0].plot(price_data.index, price_data['ADA'], label='ADA', alpha=0.8)
    axes[0, 0].plot(price_data.index, price_data['LINK']/50, label='LINK/50', alpha=0.8)
    
    # Mark trades
    if trades:
        trade_times = [t['timestamp'] for t in trades]
        trade_prices_ada = [price_data.loc[t['timestamp'], 'ADA'] for t in trades if t['timestamp'] in price_data.index]
        trade_actions = [t['action'] for t in trades if t['timestamp'] in price_data.index]
        
        for i, (time, price, action) in enumerate(zip(trade_times, trade_prices_ada, trade_actions)):
            color = 'green' if action == 'long_spread' else 'red' if action == 'short_spread' else 'black'
            marker = '^' if action == 'long_spread' else 'v' if action == 'short_spread' else 'x'
            axes[0, 0].scatter(time, price, color=color, marker=marker, s=50, alpha=0.7)
    
    axes[0, 0].set_title('Price Series with Trade Signals')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Spread evolution
    spread_index = price_data.index[:len(spreads)]
    axes[0, 1].plot(spread_index, spreads, 'purple', alpha=0.7)
    axes[0, 1].axhline(y=np.mean(spreads), color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Cointegration Spread')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Position over time
    position_index = price_data.index[:len(positions)]
    axes[1, 0].plot(position_index, positions, 'orange', linewidth=2, drawstyle='steps-post')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].set_title('Position Over Time')
    axes[1, 0].set_ylabel('Position Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Trade performance
    if trades:
        trade_profits = [t['profit'] for t in trades]
        cumulative_profit = np.cumsum(trade_profits)
        trade_numbers = range(1, len(trades) + 1)
        
        axes[1, 1].plot(trade_numbers, cumulative_profit, 'g-', linewidth=2, marker='o', markersize=4)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Cumulative P&L')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('Cumulative Profit')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Trades Executed', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Cumulative P&L')
    
    plt.tight_layout()
    plt.savefig('results/plots/hf_quick_demo.png', dpi=300, bbox_inches='tight')
    print("Quick demo plots saved: results/plots/hf_quick_demo.png")
    plt.show()


if __name__ == "__main__":
    os.makedirs('results/plots', exist_ok=True)
    
    try:
        hf_trader, trades = main()
        print("Quick high-frequency demo completed successfully!")
        
    except Exception as e:
        print(f"Error in quick HF demo: {e}")
        import traceback
        traceback.print_exc()
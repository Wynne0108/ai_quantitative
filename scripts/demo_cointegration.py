"""
Cointegration Analysis and Pairs Trading Demo Script

This script demonstrates the Stage 2.1 cointegration analysis capabilities including:
1. Multi-asset cointegration testing
2. Statistical relationship analysis  
3. Pairs trading strategy development
4. Performance evaluation and visualization

The script shows how to identify cointegrated pairs among cryptocurrencies
and implement a complete pairs trading strategy with risk management.

Usage: python scripts/demo_cointegration.py

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

# Import custom modules
from data_collection.binance_api import BinanceDataCollector
from models.intermediate.cointegration import CointegrationAnalyzer
from models.intermediate.statistical_tests import StatisticalTests
from models.intermediate.pairs_trading import PairsTradingStrategy


def main():
    """
    Main function demonstrating comprehensive cointegration analysis
    """
    print("=== Cryptocurrency Cointegration Analysis & Pairs Trading Demo ===")
    
    # Step 1: Collect multi-asset cryptocurrency data
    print("\n1. Collecting Multi-Asset Cryptocurrency Data...")
    crypto_data = collect_crypto_data()
    
    if crypto_data.empty:
        print("Failed to collect data, using simulated data instead")
        crypto_data = generate_sample_data()
    else:
        print(f"Successfully collected data for analysis")
        print(f"Data shape: {crypto_data.shape}")
        print(f"Assets: {list(crypto_data.columns)}")
        print(f"Date range: {crypto_data.index[0]} to {crypto_data.index[-1]}")
    
    # Step 2: Comprehensive cointegration analysis
    print("\n2. Running Comprehensive Cointegration Analysis...")
    coint_analyzer = CointegrationAnalyzer(
        price_data=crypto_data,
        lookback_period=200,
        confidence_level=0.05
    )
    
    # Test pairwise relationships
    pairwise_results = coint_analyzer.analyze_pairs_cointegration()
    
    # Find best cointegrated pairs
    cointegrated_pairs = [(k, v) for k, v in pairwise_results.items() 
                         if v['is_cointegrated']]
    
    print(f"Found {len(cointegrated_pairs)} cointegrated pairs:")
    for pair_name, pair_info in cointegrated_pairs:
        print(f"  {pair_name}: p-value = {pair_info['min_p_value']:.6f}")
    
    # Johansen multivariate test
    print("\n3. Running Johansen Multivariate Cointegration Test...")
    johansen_results = coint_analyzer.johansen_test(det_order=0, k_ar_diff=1)
    
    print(f"Johansen Test Results:")
    print(f"  Number of assets: {johansen_results['n_assets']}")
    print(f"  Cointegrating relationships (trace): {johansen_results['n_cointegrating_trace']}")
    print(f"  Cointegrating relationships (eigenvalue): {johansen_results['n_cointegrating_eigen']}")
    
    # Step 3: Statistical tests on individual series
    print("\n4. Running Statistical Diagnostics on Individual Assets...")
    stat_tests = StatisticalTests()
    
    # Test each asset
    for asset in crypto_data.columns[:3]:  # Test first 3 assets
        print(f"\n   Testing {asset}:")
        diagnostics = stat_tests.comprehensive_diagnostics(
            crypto_data[asset], 
            plot=False
        )
        
        summary = diagnostics['summary']
        print(f"     Data Quality: {summary['data_quality']}")
        print(f"     Main Characteristics: {', '.join(summary['main_characteristics'])}")
        print(f"     Recommendations: {', '.join(summary['modeling_recommendations'][:2])}")
    
    # Step 4: Implement pairs trading strategy
    if cointegrated_pairs:
        print(f"\n5. Implementing Pairs Trading Strategy...")
        
        # Select best pair (lowest p-value)
        best_pair = min(cointegrated_pairs, key=lambda x: x[1]['min_p_value'])
        pair_name, pair_info = best_pair
        
        # Extract asset names
        asset1, asset2 = pair_info['asset1_name'], pair_info['asset2_name']
        print(f"Selected pair: {asset1} vs {asset2}")
        print(f"Cointegration p-value: {pair_info['min_p_value']:.6f}")
        
        # Initialize pairs trading strategy
        pairs_strategy = PairsTradingStrategy(
            price_data=crypto_data,
            pair_assets=(asset1, asset2),
            lookback_period=200
        )
        
        # Configure trading parameters
        pairs_strategy.set_trading_parameters(
            entry_threshold=2.0,
            exit_threshold=0.5,
            stop_loss_threshold=4.0,
            lookback_zscore=60,
            transaction_cost=0.001,
            position_size=1.0
        )
        
        # Analyze pair relationship
        pair_analysis = pairs_strategy.analyze_pair_relationship()
        
        # Generate trading signals using multiple methods
        print(f"\n6. Generating Trading Signals...")
        
        methods = ['zscore', 'bollinger', 'percentile']
        strategy_results = {}
        
        for method in methods:
            print(f"   Testing {method} method...")
            
            # Generate signals
            signals = pairs_strategy.generate_trading_signals(
                method=method, 
                adaptive_thresholds=False
            )
            
            # Calculate performance
            performance = pairs_strategy.calculate_performance_metrics()
            
            strategy_results[method] = {
                'signals': signals,
                'performance': performance
            }
            
            print(f"     Total Return: {performance['total_return']*100:.2f}%")
            print(f"     Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            print(f"     Max Drawdown: {performance['max_drawdown']*100:.2f}%")
            print(f"     Number of Trades: {performance['number_of_trades']}")
        
        # Step 5: Compare strategies and select best
        print(f"\n7. Strategy Comparison and Selection...")
        
        best_strategy = max(strategy_results.items(), 
                           key=lambda x: x[1]['performance']['sharpe_ratio'])
        best_method, best_results = best_strategy
        
        print(f"Best performing method: {best_method}")
        print(f"Performance summary:")
        perf = best_results['performance']
        print(f"  Annual Return: {perf['annualized_return']*100:.2f}%")
        print(f"  Volatility: {perf['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        print(f"  Calmar Ratio: {perf['calmar_ratio']:.2f}")
        print(f"  Max Drawdown: {perf['max_drawdown']*100:.2f}%")
        print(f"  Win Rate: {perf['win_rate']*100:.1f}%")
        
        # Reinitialize strategy with best method
        pairs_strategy.generate_trading_signals(method=best_method)
        pairs_strategy.calculate_performance_metrics()
        
        # Step 6: Advanced analysis
        print(f"\n8. Advanced Cointegration Analysis...")
        
        # VECM estimation
        if johansen_results['n_cointegrating_trace'] > 0:
            print("   Estimating Vector Error Correction Model (VECM)...")
            vecm_results = coint_analyzer.estimate_vecm(
                n_coint_relations=min(johansen_results['n_cointegrating_trace'], 2),
                k_ar_diff=1
            )
            
            if 'error' not in vecm_results:
                print(f"     VECM estimation successful")
                print(f"     Log Likelihood: {vecm_results['log_likelihood']:.2f}")
                print(f"     AIC: {vecm_results['aic']:.2f}")
                print(f"     BIC: {vecm_results['bic']:.2f}")
            else:
                print(f"     VECM estimation failed: {vecm_results['error']}")
        
        # Step 7: Risk analysis
        print(f"\n9. Risk Analysis...")
        
        # Portfolio-level risk metrics
        signals = pairs_strategy.signals_data
        if 'net_return' in signals.columns:
            returns = signals['net_return'].dropna()
            
            if len(returns) > 0:
                # Rolling risk metrics
                rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
                rolling_sharpe = (returns.rolling(window=30).mean() / 
                                returns.rolling(window=30).std()) * np.sqrt(252)
                
                print(f"   Current volatility (30-day): {rolling_vol.iloc[-1]*100:.2f}%")
                print(f"   Current Sharpe ratio (30-day): {rolling_sharpe.iloc[-1]:.2f}")
                
                # Tail risk measures
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
                cvar_95 = returns[returns <= var_95].mean()
                
                print(f"   VaR (95%): {var_95*100:.2f}%")
                print(f"   VaR (99%): {var_99*100:.2f}%")
                print(f"   CVaR (95%): {cvar_95*100:.2f}%")
        
        # Step 8: Visualization
        print(f"\n10. Creating Visualizations...")
        
        # Cointegration analysis plots
        coint_analyzer.plot_cointegration_analysis(pair_name)
        
        # Strategy performance plots
        pairs_strategy.plot_strategy_analysis(
            figsize=(16, 12),
            save_path='results/plots/pairs_trading_analysis.png'
        )
        
        # Step 9: Save results
        print(f"\n11. Saving Results...")
        save_cointegration_results(
            coint_analyzer, 
            pairs_strategy, 
            pairwise_results, 
            johansen_results,
            strategy_results
        )
        
        # Step 10: Generate reports
        print(f"\n12. Generating Reports...")
        
        # Cointegration summary report
        coint_report = coint_analyzer.get_summary_report()
        print("\n" + "="*50)
        print("COINTEGRATION ANALYSIS SUMMARY")
        print("="*50)
        print(coint_report)
        
        # Strategy report
        strategy_report = pairs_strategy.get_strategy_report()
        print("\n" + "="*50)
        print("PAIRS TRADING STRATEGY REPORT")
        print("="*50)
        print(strategy_report)
        
        print(f"\nCointegration Analysis and Pairs Trading Demo Complete!")
        print(f"Results saved to: results/model_outputs/")
        print(f"Plots saved to: results/plots/")
        
        return coint_analyzer, pairs_strategy, strategy_results
    
    else:
        print("\nNo cointegrated pairs found. Cannot proceed with pairs trading strategy.")
        print("This might indicate:")
        print("1. Assets are not cointegrated in this time period")
        print("2. Need longer data history for reliable testing")
        print("3. Different asset selection might be required")
        
        return coint_analyzer, None, None


def collect_crypto_data():
    """
    Collect multi-asset cryptocurrency data for cointegration analysis
    """
    collector = BinanceDataCollector(verify_ssl=False)
    
    # Major crypto pairs for cointegration analysis
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
    
    try:
        crypto_data = pd.DataFrame()
        
        for symbol in symbols:
            print(f"  Collecting {symbol}...")
            
            data = collector.get_multiple_periods_data(
                symbol=symbol,
                interval='4h',
                days=180,  # 6 months of data
                market_type='spot'
            )
            
            if not data.empty:
                # Use close prices for cointegration analysis
                crypto_data[symbol.replace('USDT', '')] = data['close']
        
        if not crypto_data.empty:
            # Remove any rows with NaN values
            crypto_data = crypto_data.dropna()
            
            # Save raw data
            os.makedirs('results/data', exist_ok=True)
            crypto_data.to_csv('results/data/crypto_cointegration_data.csv')
            print(f"  Data saved to: results/data/crypto_cointegration_data.csv")
        
        return crypto_data
        
    except Exception as e:
        print(f"Error collecting crypto data: {e}")
        return pd.DataFrame()


def generate_sample_data():
    """
    Generate sample cryptocurrency data for demonstration
    """
    print("  Generating sample data...")
    
    np.random.seed(42)
    n_days = 180
    dates = pd.date_range('2024-01-01', periods=n_days*6, freq='4H')
    
    # Generate correlated price series with some cointegration
    # BTC as base asset
    btc_returns = np.random.normal(0.0008, 0.025, len(dates))
    btc_prices = 50000 * np.exp(np.cumsum(btc_returns))
    
    # ETH with high correlation to BTC (cointegrated)
    eth_noise = np.random.normal(0, 0.01, len(dates))
    eth_returns = 0.8 * btc_returns + eth_noise
    eth_prices = 3000 * np.exp(np.cumsum(eth_returns))
    
    # ADA with moderate correlation
    ada_noise = np.random.normal(0, 0.02, len(dates))
    ada_returns = 0.4 * btc_returns + ada_noise
    ada_prices = 1.2 * np.exp(np.cumsum(ada_returns))
    
    # DOT with some correlation
    dot_noise = np.random.normal(0, 0.03, len(dates))
    dot_returns = 0.3 * btc_returns + dot_noise
    dot_prices = 25 * np.exp(np.cumsum(dot_returns))
    
    # LINK with lower correlation
    link_noise = np.random.normal(0, 0.035, len(dates))
    link_returns = 0.2 * btc_returns + link_noise
    link_prices = 15 * np.exp(np.cumsum(link_returns))
    
    # Create DataFrame
    crypto_data = pd.DataFrame({
        'BTC': btc_prices,
        'ETH': eth_prices,
        'ADA': ada_prices,
        'DOT': dot_prices,
        'LINK': link_prices
    }, index=dates)
    
    # Save sample data
    os.makedirs('results/data', exist_ok=True)
    crypto_data.to_csv('results/data/sample_crypto_data.csv')
    
    return crypto_data


def save_cointegration_results(coint_analyzer, pairs_strategy, pairwise_results, 
                              johansen_results, strategy_results):
    """
    Save comprehensive analysis results to files
    """
    os.makedirs('results/model_outputs', exist_ok=True)
    
    # Save pairwise cointegration results
    pairwise_summary = []
    for pair_name, pair_info in pairwise_results.items():
        pairwise_summary.append({
            'pair': pair_name,
            'asset1': pair_info['asset1_name'],
            'asset2': pair_info['asset2_name'],
            'is_cointegrated': pair_info['is_cointegrated'],
            'p_value': pair_info['min_p_value'],
            'best_direction': pair_info['best_direction']
        })
    
    pairwise_df = pd.DataFrame(pairwise_summary)
    pairwise_df.to_csv('results/model_outputs/pairwise_cointegration_results.csv', index=False)
    
    # Save Johansen results
    johansen_summary = {
        'analysis_date': datetime.now().isoformat(),
        'n_assets': johansen_results['n_assets'],
        'asset_names': johansen_results['asset_names'],
        'n_cointegrating_trace': johansen_results['n_cointegrating_trace'],
        'n_cointegrating_eigen': johansen_results['n_cointegrating_eigen']
    }
    
    # Only add these fields if they exist (not in error case)
    if 'eigenvalues' in johansen_results:
        johansen_summary.update({
            'eigenvalues': johansen_results['eigenvalues'].tolist(),
            'trace_statistics': johansen_results['trace_statistics'].tolist(),
            'eigenvalue_statistics': johansen_results['eigenvalue_statistics'].tolist()
        })
    
    if 'error' in johansen_results:
        johansen_summary['error'] = johansen_results['error']
    
    import json
    with open('results/model_outputs/johansen_test_results.json', 'w') as f:
        json.dump(johansen_summary, f, indent=2)
    
    # Save strategy comparison results
    if strategy_results:
        strategy_comparison = []
        for method, results in strategy_results.items():
            perf = results['performance']
            strategy_comparison.append({
                'method': method,
                'total_return': perf['total_return'],
                'annualized_return': perf['annualized_return'],
                'volatility': perf['volatility'],
                'sharpe_ratio': perf['sharpe_ratio'],
                'max_drawdown': perf['max_drawdown'],
                'number_of_trades': perf['number_of_trades'],
                'win_rate': perf['win_rate']
            })
        
        strategy_df = pd.DataFrame(strategy_comparison)
        strategy_df.to_csv('results/model_outputs/strategy_comparison.csv', index=False)
    
    # Save detailed strategy results
    if pairs_strategy and pairs_strategy.signals_data is not None:
        signals_data = pairs_strategy.signals_data.copy()
        signals_data.to_csv('results/model_outputs/pairs_trading_signals.csv')
    
    print("All results saved successfully!")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/model_outputs', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('default')
    plt.rcParams['figure.max_open_warning'] = 0
    
    try:
        analyzer, strategy, results = main()
        
        if strategy is not None:
            print(f"\n=== DEMO COMPLETED SUCCESSFULLY ===")
            print(f"Cointegration analysis completed with pairs trading strategy")
            print(f"Best performing strategy method identified and implemented")
            print(f"Results and visualizations saved to results/ directory")
        else:
            print(f"\n=== DEMO COMPLETED (No Cointegrated Pairs Found) ===")
            print(f"Cointegration analysis completed but no suitable pairs found")
            print(f"Try with different assets or longer time periods")
            
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
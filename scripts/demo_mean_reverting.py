#!/usr/bin/env python3
"""
均值回歸 GBM 模型演示腳本

這個腳本演示均值回歸 GBM 模型的完整功能：
1. 數據載入和預處理
2. 參數估計（MLE, OLS, Moments）
3. 均值回歸強度分析
4. Monte Carlo 模擬
5. 價格預測和風險分析
6. 與標準 GBM 比較

Usage: python scripts/demo_mean_reverting.py

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style for better plots
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False

# Import model classes
from models.foundations.gbm_mean_reverting import MeanRevertingGBMModel
from models.foundations.gbm_base import GBMModel
from data_collection.binance_api import BinanceDataCollector

def load_sample_data():
    """Load sample data"""
    print("Loading BTC price data...")
    
    collector = BinanceDataCollector(verify_ssl=False)
    
    try:
        # Get 270 days of 4-hour data to observe mean reversion
        btc_data = collector.get_multiple_periods_data(
            symbol='BTCUSDT',
            interval='4h',
            days=270,
            market_type='spot'
        )
        
        if btc_data.empty:
            print("   Failed to get BTC data, using simulated data")
            btc_data = generate_mean_reverting_data()
        else:
            print(f"   Successfully loaded {len(btc_data)} data points")
            print(f"   Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
            print(f"   Price range: ${btc_data['close'].min():,.2f} - ${btc_data['close'].max():,.2f}")
            
            # Save data
            collector.save_data_to_csv(btc_data, 'BTC_270days_4h_mean_reverting.csv')
            
        return btc_data
        
    except Exception as e:
        print(f"   API request failed: {e}")
        print("   Using simulated data...")
        return generate_mean_reverting_data()

def generate_mean_reverting_data(start_price=50000, days=270):
    """
    Generate sample BTC data with mean reversion characteristics
    """
    np.random.seed(42)
    
    periods = days * 6  # 4-hour intervals
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='4H')
    n = len(dates)
    
    # Mean reversion parameters
    alpha = 1.5  # Mean reversion speed
    mu = np.log(55000)  # Long-term mean (log price)
    sigma = 0.05  # Volatility per period
    dt = 4/(365*24)  # 4-hour intervals in years
    
    # Simulate log price process (OU process)
    log_prices = [np.log(start_price)]  # Starting price
    
    for i in range(1, n):
        # OU process
        log_price_new = (log_prices[-1] + 
                       alpha * (mu - log_prices[-1]) * dt + 
                       sigma * np.sqrt(dt) * np.random.normal())
        log_prices.append(log_price_new)
    
    prices = np.exp(log_prices)
    
    # Create OHLCV data
    close_prices = prices
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, len(close_prices))))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, len(close_prices))))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = start_price
    volumes = np.random.lognormal(8, 0.5, len(close_prices))
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    print(f"   Generated {len(data)} synthetic mean-reverting data points")
    return data

def compare_estimation_methods(model):
    """比較不同的參數估計方法"""
    print("\nComparing Parameter Estimation Methods...")
    
    methods = ['moments', 'ols', 'mle']
    results = {}
    
    for method in methods:
        print(f"\nEstimating with {method.upper()} method:")
        try:
            # Reinitialize model
            temp_model = MeanRevertingGBMModel(model.data, freq=model.freq)
            params = temp_model.estimate_parameters(method=method)
            results[method] = params
        except Exception as e:
            print(f"   Error with {method}: {e}")
            results[method] = None
    
    # Compare results
    print("\nParameter Comparison:")
    print("=" * 60)
    print(f"{'Method':<10} {'Alpha':<8} {'Long-term':<12} {'Volatility':<12} {'Half-life':<10}")
    print(f"{'':10} {'':8} {'Price ($)':<12} {'(%)':<12} {'(days)':<10}")
    print("-" * 60)
    
    for method, params in results.items():
        if params is not None:
            print(f"{method.upper():<10} {params['alpha']:<8.4f} "
                  f"{params['long_term_price']:<12,.0f} "
                  f"{params['sigma_annual']*100:<12.2f} "
                  f"{params['half_life_days']:<10.1f}")
        else:
            print(f"{method.upper():<10} {'Failed':<8} {'':12} {'':12} {'':10}")
    
    return results

def trading_signal_analysis(model):
    """Generate trading signals based on mean reversion"""
    print("\nGenerating Trading Signals...")
    
    # 獲取均值回歸分析
    reversion_analysis = model.analyze_mean_reversion_strength()
    
    current_price = model.prices.iloc[-1]
    long_term_price = np.exp(model.mu)
    long_term_vol = model.sigma / np.sqrt(2 * model.alpha)
    
    # 計算信號強度
    deviation = abs(np.log(current_price) - model.mu)
    signal_strength = min(deviation / long_term_vol, 3.0)  # 限制在 3 sigma
    
    # 生成交易建議
    if np.log(current_price) > model.mu + 2 * long_term_vol:
        signal = "STRONG SELL"
        confidence = min(signal_strength * 33, 95)
    elif np.log(current_price) > model.mu + long_term_vol:
        signal = "SELL"
        confidence = min(signal_strength * 25, 80)
    elif np.log(current_price) < model.mu - 2 * long_term_vol:
        signal = "STRONG BUY"
        confidence = min(signal_strength * 33, 95)
    elif np.log(current_price) < model.mu - long_term_vol:
        signal = "BUY"
        confidence = min(signal_strength * 25, 80)
    else:
        signal = "HOLD"
        confidence = 50
    
    # 預期回歸時間和目標價格
    time_to_mean_50 = model.half_life_days if hasattr(model, 'half_life_days') else reversion_analysis['half_life_days']
    time_to_mean_90 = reversion_analysis['reversion_90_percent_days']
    
    print("Trading Signal Analysis:")
    print("=" * 50)
    print(f"Current Price: ${current_price:,.2f}")
    print(f"Long-term Fair Value: ${long_term_price:,.2f}")
    print(f"Position: {reversion_analysis['current_position']}")
    print(f"Signal: {signal}")
    print(f"Confidence: {confidence:.0f}%")
    print(f"Signal Strength: {signal_strength:.2f}σ")
    print(f"Expected time to 50% reversion: {time_to_mean_50:.0f} days")
    print(f"Expected time to 90% reversion: {time_to_mean_90:.0f} days")
    
    # 價格目標
    targets = reversion_analysis['price_bands']
    print(f"\nPrice Targets:")
    print(f"   Strong Resistance: ${targets['upper_2std']:,.2f}")
    print(f"   Resistance: ${targets['upper_1std']:,.2f}")
    print(f"   Fair Value: ${targets['mean']:,.2f}")
    print(f"   Support: ${targets['lower_1std']:,.2f}")
    print(f"   Strong Support: ${targets['lower_2std']:,.2f}")
    
    return {
        'signal': signal,
        'confidence': confidence,
        'signal_strength': signal_strength,
        'targets': targets,
        'reversion_times': {
            '50_percent': time_to_mean_50,
            '90_percent': time_to_mean_90
        }
    }

def compare_with_standard_gbm(mean_rev_model, data):
    """Compare with standard GBM model"""
    print("\nComparing Mean-Reverting GBM vs Standard GBM...")
    
    # 創建標準 GBM 模型
    standard_gbm = GBMModel(data, freq='4H')
    gbm_params = standard_gbm.estimate_parameters()
    
    # 預測比較
    horizons = [30, 90, 180]
    
    print("\nForecast Comparison:")
    print("=" * 80)
    print(f"{'Horizon':<8} {'Mean-Reverting':<20} {'Standard GBM':<20} {'Difference':<15}")
    print(f"{'(days)':<8} {'Expected Price':<20} {'Expected Price':<20} {'(%)':<15}")
    print("-" * 80)
    
    forecast_comparison = {}
    
    for horizon in horizons:
        mr_forecast = mean_rev_model.forecast_price(time_horizon=horizon)
        gbm_forecast = standard_gbm.forecast_price(time_horizon=horizon)
        
        mr_price = mr_forecast['expected_price']
        gbm_price = gbm_forecast['expected_price']
        diff_pct = ((mr_price - gbm_price) / gbm_price) * 100
        
        forecast_comparison[horizon] = {
            'mean_reverting': mr_price,
            'standard_gbm': gbm_price,
            'difference_pct': diff_pct
        }
        
        print(f"{horizon:<8} ${mr_price:<19,.0f} ${gbm_price:<19,.0f} {diff_pct:<+14.1f}")
    
    # 風險比較
    print("\nRisk Comparison (7-day VaR):")
    mr_risk = mean_rev_model.calculate_var_cvar(time_horizon=7)
    gbm_risk = standard_gbm.calculate_var_cvar(time_horizon=7)
    
    print("=" * 60)
    print(f"{'Model':<20} {'95% VaR':<15} {'99% VaR':<15}")
    print("-" * 60)
    print(f"{'Mean-Reverting':<20} ${mr_risk['VaR_95']:<14,.0f} ${mr_risk['VaR_99']:<14,.0f}")
    print(f"{'Standard GBM':<20} ${gbm_risk['VaR_95']:<14,.0f} ${gbm_risk['VaR_99']:<14,.0f}")
    
    return {
        'forecasts': forecast_comparison,
        'risk_metrics': {
            'mean_reverting': mr_risk,
            'standard_gbm': gbm_risk
        }
    }

def main():
    """Main demo function"""
    print("=== Mean-Reverting GBM Model Demo ===")
    print("=" * 60)
    
    try:
        # 1. 載入數據
        data = load_sample_data()
        
        # 2. Create and configure model
        print("\n2. Initializing Mean-Reverting GBM Model...")
        model = MeanRevertingGBMModel(data, price_column='close', freq='4H')
        
        # 3. Compare parameter estimation methods
        estimation_comparison = compare_estimation_methods(model)
        
        # Use best method (usually MLE)
        print("\n3. Using MLE estimation for final analysis...")
        params = model.estimate_parameters(method='mle')
        
        # 4. Mean reversion strength analysis
        print("\n4. Analyzing Mean Reversion Strength...")
        reversion_analysis = model.analyze_mean_reversion_strength()
        
        # 5. Trading signal analysis
        trading_signals = trading_signal_analysis(model)
        
        # 6. Monte Carlo simulation
        print("\n5. Running Monte Carlo Simulation...")
        simulation = model.simulate_paths(n_paths=1000, n_steps=126)  # Smaller simulation for demo
        
        # 7. Price forecasts
        print("\n6. Generating Price Forecasts...")
        forecast_30d = model.forecast_price(time_horizon=30)
        forecast_90d = model.forecast_price(time_horizon=90)
        forecast_180d = model.forecast_price(time_horizon=180)
        
        # 8. Risk analysis
        print("\n7. Calculating Risk Metrics...")
        risk_metrics = model.calculate_var_cvar(
            portfolio_value=100000,
            confidence_levels=[0.95, 0.99],
            time_horizon=7
        )
        
        # 9. Compare with standard GBM
        comparison = compare_with_standard_gbm(model, data)
        
        # 10. Create analysis charts
        print("\n8. Creating Analysis Charts...")
        model.plot_analysis()
        
        # 11. Save results summary
        print("\n9. Saving Results Summary...")
        
        summary = {
            'model': 'Mean-Reverting GBM',
            'data_period': f"{data.index[0]} to {data.index[-1]}",
            'parameters': params,
            'estimation_comparison': estimation_comparison,
            'mean_reversion_analysis': reversion_analysis,
            'trading_signals': trading_signals,
            'forecasts': {
                '30_day': forecast_30d,
                '90_day': forecast_90d,
                '180_day': forecast_180d
            },
            'risk_metrics': risk_metrics,
            'model_comparison': comparison,
            'simulation_summary': {
                'n_paths': simulation['n_paths'],
                'n_steps': simulation['n_steps'],
                'final_price_mean': float(np.mean(simulation['final_prices'])),
                'final_price_std': float(np.std(simulation['final_prices'])),
                'mean_reversion_stats': simulation['mean_reversion_stats']
            }
        }
        
        # 保存為 JSON
        import json
        os.makedirs('results/models', exist_ok=True)
        with open('results/models/mean_reverting_gbm_results.json', 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        print("\nMean-Reverting GBM Analysis Complete!")
        print("=" * 60)
        print("Results saved to:")
        print("   Charts: results/plots/mean_reverting_gbm_analysis.png")
        print("   Summary: results/models/mean_reverting_gbm_results.json")
        
        # Key results summary
        current_price = data['close'].iloc[-1]
        print(f"\nKey Results Summary:")
        print(f"   Current BTC Price: ${current_price:,.2f}")
        print(f"   Long-term Fair Value: ${reversion_analysis['long_term_price']:,.2f}")
        print(f"   Mean Reversion Strength: {reversion_analysis['strength_classification']}")
        print(f"   Current Position: {reversion_analysis['current_position']}")
        print(f"   Half-life to Mean: {reversion_analysis['half_life_days']:.0f} days")
        print(f"   Trading Signal: {trading_signals['signal']} ({trading_signals['confidence']:.0f}% confidence)")
        print(f"   30-day Expected Price: ${forecast_30d['expected_price']:,.2f}")
        print(f"   90-day Expected Price: ${forecast_90d['expected_price']:,.2f}")
        print(f"   95% VaR (7-day): ${risk_metrics['VaR_95']:,.0f}")
        
        # Practical insights
        print(f"\nPractical Insights:")
        if reversion_analysis['current_position'] in ['Overvalued', 'Extremely Overvalued']:
            print("   Current price is above fair value - consider taking profits")
        elif reversion_analysis['current_position'] in ['Undervalued', 'Extremely Undervalued']:
            print("   Current price is below fair value - potential buying opportunity")
        else:
            print("   Current price is near fair value - market appears balanced")
        
        print(f"   Expected reversion timeline: {reversion_analysis['half_life_days']:.0f} days for 50% correction")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""
基礎 GBM 模型演示腳本

這個腳本展示如何使用基礎 GBM 模型進行：
1. 參數估計和模型擬合
2. Monte Carlo 價格路徑模擬  
3. 價格預測和信心區間
4. 風險指標計算 (VaR/CVaR)
5. 模型診斷和視覺化分析

Usage: python scripts/demo_gbm_basic.py

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
from scipy import stats

# Set matplotlib style for better plots
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False

# Import custom modules
from data_collection.binance_api import BinanceDataCollector
from models.foundations.gbm_base import GBMModel

def main():
    """
    Main function to run basic GBM analysis on BTC data
    """
    print("=== BTC Geometric Brownian Motion (GBM) Analysis ===")
    
    # Step 1: Collect BTC data
    print("\n1. Collecting BTC Data...")
    collector = BinanceDataCollector(verify_ssl=False)
    
    try:
        # Get 120 days of 4-hour BTC data for GBM analysis
        btc_data = collector.get_multiple_periods_data(
            symbol='BTCUSDT',
            interval='4h',
            days=120,
            market_type='spot'
        )
        
        if btc_data.empty:
            print("Failed to get BTC data, using simulated data")
            btc_data = generate_sample_data()
        else:
            print(f"Successfully collected {len(btc_data)} data points")
            print(f"Data period: {btc_data.index[0]} to {btc_data.index[-1]}")
            
            # Save data
            collector.save_data_to_csv(btc_data, 'BTC_120days_4h_gbm.csv')
            
    except Exception as e:
        print(f"API request failed: {e}")
        print("Using simulated data...")
        btc_data = generate_sample_data()
    
    # Step 2: Initialize GBM model
    print("\n2. Initializing GBM Model...")
    gbm_model = GBMModel(btc_data, price_column='close', freq='4h')
    
    # Step 3: Estimate parameters
    print("\n3. Estimating GBM Parameters...")
    parameters = gbm_model.estimate_parameters(method='mle')
    
    # Step 4: Analyze model fit
    print("\n4. Analyzing Model Fit...")
    fit_analysis = gbm_model.analyze_model_fit()
    
    # Step 5: Generate price forecasts
    print("\n5. Generating Price Forecasts...")
    forecast_30d = gbm_model.forecast_price(time_horizon=30, confidence_levels=[0.95, 0.99])
    forecast_90d = gbm_model.forecast_price(time_horizon=90, confidence_levels=[0.95, 0.99])
    
    # Step 6: Monte Carlo simulation
    print("\n6. Running Monte Carlo Simulation...")
    simulation = gbm_model.simulate_paths(
        n_paths=10000, 
        n_steps=252,  # 1 year daily
        random_seed=42
    )
    
    # Step 7: Calculate risk metrics
    print("\n7. Calculating Risk Metrics...")
    risk_1d = gbm_model.calculate_var_cvar(
        portfolio_value=100000,
        confidence_levels=[0.95, 0.99],
        time_horizon=1
    )
    
    risk_30d = gbm_model.calculate_var_cvar(
        portfolio_value=100000,
        confidence_levels=[0.95, 0.99], 
        time_horizon=30
    )
    
    # Step 8: Display comprehensive results
    print("\n8. Comprehensive Analysis Results...")
    
    current_price = btc_data['close'].iloc[-1]
    mu_annual = parameters['mu_annual']
    sigma_annual = parameters['sigma_annual']
    
    print(f"\nGBM Model Results:")
    print(f"   Current BTC Price: ${current_price:,.2f}")
    print(f"   Annual Drift Rate (mu): {mu_annual:.4f} ({mu_annual*100:+.2f}%)")
    print(f"   Annual Volatility (sigma): {sigma_annual:.4f} ({sigma_annual*100:.2f}%)")
    
    print(f"\nPrice Forecasts:")
    print(f"   30-day expected price: ${forecast_30d['expected_price']:,.2f}")
    print(f"   30-day change: {((forecast_30d['expected_price']/current_price)-1)*100:+.2f}%")
    print(f"   30-day 95% CI: ${forecast_30d['lower_95']:,.2f} - ${forecast_30d['upper_95']:,.2f}")
    
    print(f"   90-day expected price: ${forecast_90d['expected_price']:,.2f}")
    print(f"   90-day change: {((forecast_90d['expected_price']/current_price)-1)*100:+.2f}%")
    print(f"   90-day 95% CI: ${forecast_90d['lower_95']:,.2f} - ${forecast_90d['upper_95']:,.2f}")
    
    print(f"\nMonte Carlo Results (1 Year):")
    final_prices = simulation['final_prices']
    print(f"   Mean final price: ${np.mean(final_prices):,.2f}")
    print(f"   Median final price: ${np.median(final_prices):,.2f}")
    print(f"   Price range (5%-95%): ${np.percentile(final_prices, 5):,.2f} - ${np.percentile(final_prices, 95):,.2f}")
    
    returns_1yr = (final_prices - current_price) / current_price
    print(f"   Mean return: {np.mean(returns_1yr)*100:+.2f}%")
    print(f"   Return volatility: {np.std(returns_1yr)*100:.2f}%")
    print(f"   Probability of profit: {(returns_1yr > 0).mean()*100:.1f}%")
    
    print(f"\nRisk Analysis ($100,000 portfolio):")
    print(f"   1-day 95% VaR: ${risk_1d['VaR_95']:,.0f}")
    print(f"   1-day 99% VaR: ${risk_1d['VaR_99']:,.0f}")
    print(f"   30-day 95% VaR: ${risk_30d['VaR_95']:,.0f}")
    print(f"   30-day 99% VaR: ${risk_30d['VaR_99']:,.0f}")
    
    # Step 9: Model quality assessment
    print(f"\nModel Quality Assessment:")
    print(f"   Normality test: {'PASS' if fit_analysis['normality_test_pass'] else 'FAIL'}")
    print(f"   Distribution fit: {'PASS' if fit_analysis['distribution_test_pass'] else 'FAIL'}")
    print(f"   Independence test: {'PASS' if fit_analysis['independence_test_pass'] else 'FAIL'}")
    
    if mu_annual > 0:
        print(f"   Model suggests positive drift (bullish trend)")
    else:
        print(f"   Model suggests negative drift (bearish trend)")
    
    vol_level = "HIGH" if sigma_annual > 1.0 else "MODERATE" if sigma_annual > 0.5 else "LOW"
    print(f"   Volatility level: {vol_level}")
    
    # Step 10: Investment insights
    print(f"\nInvestment Insights:")
    
    # Time to double
    if mu_annual > 0:
        time_to_double = np.log(2) / mu_annual
        print(f"   Time to double (at drift rate): {time_to_double:.1f} years")
    
    # Risk-adjusted return (Sharpe ratio approximation)
    risk_free_rate = 0.03  # Assume 3% risk-free rate
    sharpe_approx = (mu_annual - risk_free_rate) / sigma_annual
    print(f"   Sharpe ratio (approx): {sharpe_approx:.2f}")
    
    # Probability of specific targets
    target_2x = current_price * 2
    target_half = current_price * 0.5
    
    # 1 year probabilities
    ln_S0 = np.log(current_price)
    mean_ln_1y = ln_S0 + (mu_annual - 0.5 * sigma_annual**2) * 1
    std_ln_1y = sigma_annual * np.sqrt(1)
    
    prob_2x = 1 - stats.norm.cdf(np.log(target_2x), mean_ln_1y, std_ln_1y)
    prob_half = stats.norm.cdf(np.log(target_half), mean_ln_1y, std_ln_1y)
    
    print(f"   Probability of 2x return (1 year): {prob_2x*100:.1f}%")
    print(f"   Probability of 50% loss (1 year): {prob_half*100:.1f}%")
    
    # Step 11: Create visualizations
    print("\n9. Creating Analysis Charts...")
    gbm_model.plot_analysis()
    
    # Step 12: Save results
    print("\n10. Saving Results...")
    save_results(gbm_model, parameters, forecast_30d, forecast_90d, simulation, risk_1d, risk_30d, fit_analysis)
    
    print("Basic GBM Analysis Complete!")
    print("Charts saved: results/plots/gbm_analysis.png")
    print("Results saved: results/model_outputs/")
    
    return gbm_model, parameters, simulation

def save_results(model, params, forecast_30d, forecast_90d, simulation, risk_1d, risk_30d, fit_analysis):
    """
    Save analysis results to files
    """
    os.makedirs('results/model_outputs', exist_ok=True)
    
    # Save simulation results
    simulation_summary = pd.DataFrame({
        'Path_' + str(i): simulation['paths'][i] 
        for i in range(min(100, simulation['n_paths']))  # Save first 100 paths
    }, index=simulation['prices'].index)
    
    simulation_summary.to_csv('results/model_outputs/gbm_simulation_paths.csv')
    
    # Save forecasts
    forecast_df = pd.DataFrame({
        '30d_expected': forecast_30d['expected_price'],
        '30d_lower_95': forecast_30d['lower_95'],
        '30d_upper_95': forecast_30d['upper_95'],
        '90d_expected': forecast_90d['expected_price'],
        '90d_lower_95': forecast_90d['lower_95'],
        '90d_upper_95': forecast_90d['upper_95']
    }, index=[0])
    
    forecast_df.to_csv('results/model_outputs/gbm_price_forecasts.csv')
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            return obj
    
    # Save comprehensive summary
    summary = {
        'model': 'Geometric Brownian Motion (GBM)',
        'analysis_date': datetime.now().isoformat(),
        'data_points': len(model.prices),
        'data_period': f"{model.prices.index[0]} to {model.prices.index[-1]}",
        
        # Parameters
        'parameters': params,
        
        # Current status
        'current_price': float(model.prices.iloc[-1]),
        
        # Forecasts
        'forecasts': {
            '30_day': {
                'expected_price': float(forecast_30d['expected_price']),
                'expected_return': float((forecast_30d['expected_price']/model.prices.iloc[-1] - 1) * 100),
                'confidence_interval_95': [float(forecast_30d['lower_95']), float(forecast_30d['upper_95'])]
            },
            '90_day': {
                'expected_price': float(forecast_90d['expected_price']),
                'expected_return': float((forecast_90d['expected_price']/model.prices.iloc[-1] - 1) * 100),
                'confidence_interval_95': [float(forecast_90d['lower_95']), float(forecast_90d['upper_95'])]
            }
        },
        
        # Risk metrics
        'risk_metrics': {
            '1_day': {k: float(v) for k, v in risk_1d.items() if isinstance(v, (int, float))},
            '30_day': {k: float(v) for k, v in risk_30d.items() if isinstance(v, (int, float))}
        },
        
        # Model quality (convert numpy bool_ to Python bool)
        'model_quality': {
            k: bool(v) if isinstance(v, np.bool_) else float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in fit_analysis.items()
        },
        
        # Simulation summary
        'simulation_summary': {
            'n_paths': simulation['n_paths'],
            'n_steps': simulation['n_steps'],
            'final_price_stats': {
                'mean': float(np.mean(simulation['final_prices'])),
                'median': float(np.median(simulation['final_prices'])),
                'std': float(np.std(simulation['final_prices'])),
                'min': float(np.min(simulation['final_prices'])),
                'max': float(np.max(simulation['final_prices']))
            }
        }
    }
    
    import json
    with open('results/model_outputs/gbm_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

def generate_sample_data(start_price=50000, days=120):
    """
    Generate sample BTC data with realistic GBM characteristics
    """
    np.random.seed(42)
    
    periods = days * 6  # 4-hour intervals
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='4H')
    
    # Simulate realistic BTC GBM process
    mu = 0.8   # 80% annual drift
    sigma = 1.2  # 120% annual volatility
    dt = 1/365/6  # 4-hour intervals
    
    # Generate price path
    n = len(dates)
    random_shocks = np.random.normal(0, 1, n)
    
    log_returns = np.zeros(n)
    log_returns[0] = np.log(start_price)
    
    for t in range(1, n):
        log_returns[t] = log_returns[t-1] + (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks[t]
    
    prices = np.exp(log_returns)
    
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
    
    return data

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/model_outputs', exist_ok=True)
    
    # Run analysis
    try:
        model, params, simulation = main()
        print(f"\nGBM Analysis Successfully Completed!")
        print(f"Annual Drift: {params['mu_annual']*100:+.2f}%")
        print(f"Annual Volatility: {params['sigma_annual']*100:.2f}%")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
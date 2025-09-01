"""
進階 GBM 模型演示腳本 - 包含跳躍擴散模型

這個腳本展示如何使用進階 GBM 模型進行：
1. Merton 跳躍擴散模型分析
2. 跳躍檢測和事件影響評估  
3. 基礎 GBM vs Merton 模型比較
4. 跳躍風險的量化分析
5. 不同市場條件下的模型表現

Usage: python scripts/demo_gbm_advanced.py

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
from models.foundations.gbm_merton import MertonJumpDiffusionModel

def main():
    """
    Main function to run advanced GBM analysis with jump diffusion
    """
    print("=== Advanced GBM Analysis: Merton Jump Diffusion Model ===")
    
    # Step 1: Collect BTC data
    print("\n1. Collecting BTC Data...")
    collector = BinanceDataCollector(verify_ssl=False)
    
    try:
        # Get 180 days of 4-hour BTC data for jump analysis  
        btc_data = collector.get_multiple_periods_data(
            symbol='BTCUSDT',
            interval='4h',
            days=180,
            market_type='spot'
        )
        
        if btc_data.empty:
            print("Failed to get BTC data, using simulated data")
            btc_data = generate_jump_sample_data()
        else:
            print(f"Successfully collected {len(btc_data)} data points")
            print(f"Data period: {btc_data.index[0]} to {btc_data.index[-1]}")
            
            # Save data
            collector.save_data_to_csv(btc_data, 'BTC_180days_4h_merton.csv')
            
    except Exception as e:
        print(f"API request failed: {e}")
        print("Using simulated data...")
        btc_data = generate_jump_sample_data()
    
    # Step 2: Initialize both models for comparison
    print("\n2. Initializing Models...")
    basic_gbm = GBMModel(btc_data, price_column='close', freq='4h')
    merton_model = MertonJumpDiffusionModel(btc_data, price_column='close', freq='4h')
    
    # Step 3: Estimate parameters for both models
    print("\n3. Estimating Model Parameters...")
    
    print("   3a. Basic GBM Parameters:")
    gbm_params = basic_gbm.estimate_parameters(method='mle')
    
    print("   3b. Merton Jump Diffusion Parameters:")
    merton_params = merton_model.estimate_parameters(method='two_stage', jump_threshold=3.0)
    
    # Step 4: Jump detection and analysis
    print("\n4. Jump Detection Analysis...")
    jump_results = merton_model.detect_jumps(threshold=3.0)
    jump_impact = merton_model.analyze_jump_impact()
    
    # Step 5: Model comparison
    print("\n5. Comparing Model Performance...")
    model_comparison = compare_models(basic_gbm, merton_model)
    
    # Step 6: Generate forecasts with both models
    print("\n6. Generating Forecasts...")
    
    gbm_forecast = basic_gbm.forecast_price(time_horizon=30, confidence_levels=[0.95])
    merton_forecast = merton_model.forecast_price(time_horizon=30, confidence_levels=[0.95])
    
    # Step 7: Monte Carlo simulations
    print("\n7. Running Monte Carlo Simulations...")
    
    print("   7a. Basic GBM Simulation:")
    gbm_simulation = basic_gbm.simulate_paths(n_paths=5000, n_steps=90, random_seed=42)
    
    print("   7b. Merton Model Simulation:")
    merton_simulation = merton_model.simulate_paths(n_paths=5000, n_steps=90, random_seed=42)
    
    # Step 8: Risk analysis comparison
    print("\n8. Risk Analysis Comparison...")
    
    gbm_risk = basic_gbm.calculate_var_cvar(
        portfolio_value=100000,
        confidence_levels=[0.95, 0.99],
        time_horizon=7
    )
    
    merton_risk = merton_model.calculate_var_cvar(
        portfolio_value=100000,
        confidence_levels=[0.95, 0.99], 
        time_horizon=7
    )
    
    # Step 9: Display comprehensive results
    print("\n9. Comprehensive Analysis Results...")
    display_results(btc_data, gbm_params, merton_params, jump_results, jump_impact, 
                   model_comparison, gbm_forecast, merton_forecast, 
                   gbm_simulation, merton_simulation, gbm_risk, merton_risk)
    
    # Step 10: Create visualizations
    print("\n10. Creating Analysis Charts...")
    
    # Jump analysis charts
    merton_model.plot_jump_analysis()
    
    # Model comparison charts
    create_model_comparison_plots(basic_gbm, merton_model, gbm_simulation, merton_simulation)
    
    # Step 11: Save results
    print("\n11. Saving Results...")
    save_advanced_results(basic_gbm, merton_model, gbm_params, merton_params, 
                         jump_results, model_comparison, gbm_simulation, merton_simulation)
    
    print("Advanced GBM Analysis Complete!")
    print("Charts saved: results/plots/")
    print("Results saved: results/model_outputs/")
    
    return basic_gbm, merton_model, jump_results

def compare_models(gbm_model, merton_model):
    """
    Compare basic GBM and Merton jump diffusion models
    """
    print("Comparing GBM vs Merton Models...")
    
    # Analyze fit quality for both models
    gbm_fit = gbm_model.analyze_model_fit()
    merton_fit = merton_model.analyze_model_fit()
    
    # Calculate log-likelihood for comparison (simplified)
    gbm_returns = gbm_model.returns
    
    # GBM log-likelihood
    gbm_ll = -0.5 * len(gbm_returns) * (1 + np.log(2 * np.pi * gbm_model.sigma**2)) - \
             0.5 * np.sum((gbm_returns - gbm_model.mu)**2) / gbm_model.sigma**2
    
    # Merton model likelihood (approximation)
    continuous_returns = gbm_returns.drop(merton_model.jump_dates) if len(merton_model.jump_dates) > 0 else gbm_returns
    merton_ll = -0.5 * len(continuous_returns) * (1 + np.log(2 * np.pi * merton_model.sigma**2)) - \
                0.5 * np.sum((continuous_returns - merton_model.mu)**2) / merton_model.sigma**2
    
    # Add jump likelihood component
    if len(merton_model.jump_sizes) > 0:
        jump_ll = len(merton_model.jump_sizes) * np.log(merton_model.lambda_jump * merton_model.dt) - \
                 merton_model.lambda_jump * merton_model.dt * len(gbm_returns)
        merton_ll += jump_ll
    
    comparison = {
        'gbm_log_likelihood': gbm_ll,
        'merton_log_likelihood': merton_ll,
        'likelihood_improvement': merton_ll - gbm_ll,
        
        'gbm_normality_pass': gbm_fit['normality_test_pass'],
        'merton_normality_pass': merton_fit['normality_test_pass'],
        
        'gbm_independence_pass': gbm_fit['independence_test_pass'],
        'merton_independence_pass': merton_fit['independence_test_pass'],
        
        'jumps_detected': len(merton_model.jump_dates),
        'jump_frequency': len(merton_model.jump_dates) / len(gbm_returns),
        
        'gbm_params_count': 2,  # mu, sigma
        'merton_params_count': 5,  # mu, sigma, lambda, mu_jump, sigma_jump
        
        # Information criteria (approximation)
        'gbm_aic': -2 * gbm_ll + 2 * 2,
        'merton_aic': -2 * merton_ll + 2 * 5,
    }
    
    print("Model comparison complete")
    print(f"   Basic GBM log-likelihood: {gbm_ll:.2f}")
    print(f"   Merton model log-likelihood: {merton_ll:.2f}")
    print(f"   Likelihood improvement: {merton_ll - gbm_ll:.2f}")
    print(f"   Jumps detected: {len(merton_model.jump_dates)}")
    print(f"   Better model: {'Merton' if merton_ll > gbm_ll else 'Basic GBM'}")
    
    return comparison

def display_results(data, gbm_params, merton_params, jump_results, jump_impact, 
                   comparison, gbm_forecast, merton_forecast, 
                   gbm_sim, merton_sim, gbm_risk, merton_risk):
    """
    Display comprehensive analysis results
    """
    current_price = data['close'].iloc[-1]
    
    print(f"\nAdvanced GBM Analysis Summary")
    print(f"=" * 60)
    
    # Current market status
    print(f"\nCurrent Market Status:")
    print(f"   BTC Price: ${current_price:,.2f}")
    print(f"   Analysis Period: {data.index[0]} to {data.index[-1]}")
    print(f"   Total Observations: {len(data)}")
    
    # Model parameters comparison
    print(f"\nModel Parameters Comparison:")
    print(f"   Basic GBM:")
    print(f"     Annual Drift: {gbm_params['mu_annual']*100:+.2f}%")
    print(f"     Annual Volatility: {gbm_params['sigma_annual']*100:.2f}%")
    
    print(f"   Merton Jump Diffusion:")
    print(f"     Annual Drift: {merton_params['mu_annual']*100:+.2f}%")
    print(f"     Annual Volatility: {merton_params['sigma_annual']*100:.2f}%")
    print(f"     Jump Intensity: {merton_params['lambda_jump']:.2f} jumps/year")
    print(f"     Jump Size Mean: {merton_params['mu_jump']*100:+.2f}%")
    print(f"     Jump Size Std: {merton_params['sigma_jump']*100:.2f}%")
    
    # Jump analysis
    print(f"\nJump Analysis:")
    print(f"   Total Jumps Detected: {jump_results['n_jumps']}")
    print(f"   Jump Frequency: {jump_results['jump_frequency']*100:.2f}%")
    print(f"   Positive Jumps: {jump_results['positive_jumps']}")
    print(f"   Negative Jumps: {jump_results['negative_jumps']}")
    
    if jump_results['n_jumps'] > 0:
        print(f"   Largest Positive Jump: {np.max(jump_results['jump_sizes'])*100:+.2f}%")
        print(f"   Largest Negative Jump: {np.min(jump_results['jump_sizes'])*100:+.2f}%")
    
    # Forecast comparison
    print(f"\n30-Day Forecast Comparison:")
    gbm_change = (gbm_forecast['expected_price'] / current_price - 1) * 100
    merton_change = (merton_forecast['expected_price'] / current_price - 1) * 100
    
    print(f"   Basic GBM Expected: ${gbm_forecast['expected_price']:,.2f} ({gbm_change:+.2f}%)")
    print(f"   Merton Expected: ${merton_forecast['expected_price']:,.2f} ({merton_change:+.2f}%)")
    print(f"   Forecast Difference: {merton_change - gbm_change:+.2f}%")
    
    # Simulation comparison
    print(f"\nMonte Carlo Simulation Comparison (90 days):")
    gbm_final = gbm_sim['final_prices']
    merton_final = merton_sim['final_prices']
    
    print(f"   Basic GBM:")
    print(f"     Mean Price: ${np.mean(gbm_final):,.2f}")
    print(f"     Price Volatility: ${np.std(gbm_final):,.2f}")
    print(f"     95% Range: ${np.percentile(gbm_final, 2.5):,.2f} - ${np.percentile(gbm_final, 97.5):,.2f}")
    
    print(f"   Merton Model:")
    print(f"     Mean Price: ${np.mean(merton_final):,.2f}")
    print(f"     Price Volatility: ${np.std(merton_final):,.2f}")
    print(f"     95% Range: ${np.percentile(merton_final, 2.5):,.2f} - ${np.percentile(merton_final, 97.5):,.2f}")
    
    # Risk comparison
    print(f"\nRisk Analysis Comparison (7-day VaR):")
    print(f"   Basic GBM - 95% VaR: ${gbm_risk['VaR_95']:,.0f}")
    print(f"   Merton Model - 95% VaR: ${merton_risk['VaR_95']:,.0f}")
    print(f"   Risk Difference: ${merton_risk['VaR_95'] - gbm_risk['VaR_95']:+,.0f}")
    
    risk_increase = (merton_risk['VaR_95'] / gbm_risk['VaR_95'] - 1) * 100
    print(f"   Jump Risk Premium: {risk_increase:+.1f}%")
    
    # Model selection recommendation
    print(f"\nModel Selection Recommendation:")
    if comparison['likelihood_improvement'] > 10:
        print("   RECOMMEND: Merton Jump Diffusion Model")
        print("   Reason: Significant likelihood improvement and substantial jump activity")
    elif jump_results['n_jumps'] > len(data) * 0.05:  # More than 5% jumps
        print("   RECOMMEND: Merton Jump Diffusion Model")  
        print("   Reason: High jump frequency detected")
    else:
        print("   RECOMMEND: Basic GBM Model")
        print("   Reason: Minimal jump activity, simpler model preferred")

def create_model_comparison_plots(gbm_model, merton_model, gbm_sim, merton_sim):
    """
    Create model comparison visualization
    """
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Simulation paths comparison
    n_paths_show = min(50, gbm_sim['n_paths'])
    
    # GBM paths
    for i in range(n_paths_show):
        axes[0, 0].plot(gbm_sim['prices'].index, gbm_sim['prices'].iloc[:, i], 
                       alpha=0.3, linewidth=0.5, color='blue')
    
    gbm_mean = gbm_sim['prices'].mean(axis=1)
    axes[0, 0].plot(gbm_mean.index, gbm_mean.values, 'b-', linewidth=2, label='GBM Mean')
    axes[0, 0].set_title('Basic GBM Simulation Paths')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Merton paths  
    for i in range(n_paths_show):
        axes[0, 1].plot(merton_sim['prices'].index, merton_sim['prices'].iloc[:, i],
                       alpha=0.3, linewidth=0.5, color='red')
    
    merton_mean = merton_sim['prices'].mean(axis=1)
    axes[0, 1].plot(merton_mean.index, merton_mean.values, 'r-', linewidth=2, label='Merton Mean')
    axes[0, 1].set_title('Merton Jump Diffusion Simulation Paths')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2. Final price distributions
    axes[1, 0].hist(gbm_sim['final_prices'], bins=50, alpha=0.7, density=True, 
                   color='blue', label='GBM')
    axes[1, 0].hist(merton_sim['final_prices'], bins=50, alpha=0.7, density=True,
                   color='red', label='Merton')
    axes[1, 0].set_title('Final Price Distributions')
    axes[1, 0].set_xlabel('Final Price')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. Return distributions
    current_price = gbm_model.prices.iloc[-1]
    gbm_returns = (gbm_sim['final_prices'] - current_price) / current_price
    merton_returns = (merton_sim['final_prices'] - current_price) / current_price
    
    axes[1, 1].hist(gbm_returns, bins=50, alpha=0.7, density=True,
                   color='blue', label='GBM Returns')
    axes[1, 1].hist(merton_returns, bins=50, alpha=0.7, density=True,
                   color='red', label='Merton Returns')
    axes[1, 1].set_title('Return Distributions')
    axes[1, 1].set_xlabel('Returns')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/gbm_model_comparison.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def save_advanced_results(gbm_model, merton_model, gbm_params, merton_params, 
                         jump_results, comparison, gbm_sim, merton_sim):
    """
    Save advanced analysis results
    """
    os.makedirs('results/model_outputs', exist_ok=True)
    
    # Save jump detection results
    if len(merton_model.jump_dates) > 0:
        jump_df = pd.DataFrame({
            'jump_date': merton_model.jump_dates,
            'jump_size': merton_model.jump_sizes,
            'jump_size_pct': merton_model.jump_sizes * 100
        })
        jump_df.to_csv('results/model_outputs/detected_jumps.csv')
    
    # Save model comparison
    comparison_df = pd.DataFrame([{
        'Model': 'Basic GBM',
        'Annual_Drift': gbm_params['mu_annual'],
        'Annual_Volatility': gbm_params['sigma_annual'],
        'Parameters': 2,
        'Log_Likelihood': comparison['gbm_log_likelihood'],
        'AIC': comparison['gbm_aic']
    }, {
        'Model': 'Merton Jump Diffusion',
        'Annual_Drift': merton_params['mu_annual'],
        'Annual_Volatility': merton_params['sigma_annual'],
        'Jump_Intensity': merton_params['lambda_jump'],
        'Jump_Mean': merton_params['mu_jump'],
        'Jump_Std': merton_params['sigma_jump'],
        'Parameters': 5,
        'Log_Likelihood': comparison['merton_log_likelihood'],
        'AIC': comparison['merton_aic']
    }])
    
    comparison_df.to_csv('results/model_outputs/model_comparison.csv', index=False)
    
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
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Index):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            return obj
    
    # Save comprehensive summary
    summary = {
        'analysis_type': 'Advanced GBM with Jump Diffusion',
        'analysis_date': datetime.now().isoformat(),
        'data_points': len(gbm_model.prices),
        'data_period': f"{gbm_model.prices.index[0]} to {gbm_model.prices.index[-1]}",
        
        'gbm_parameters': convert_numpy_types(gbm_params),
        'merton_parameters': convert_numpy_types(merton_params),
        'jump_analysis': convert_numpy_types(jump_results),
        'model_comparison': convert_numpy_types(comparison),
        
        'recommendations': {
            'preferred_model': 'Merton' if comparison['likelihood_improvement'] > 10 else 'GBM',
            'jump_significance': 'High' if jump_results['jump_frequency'] > 0.05 else 'Moderate' if jump_results['jump_frequency'] > 0.02 else 'Low',
            'use_cases': {
                'gbm': 'Long-term forecasting, option pricing in stable markets',
                'merton': 'Risk management, crisis modeling, event-driven analysis'
            }
        }
    }
    
    import json
    with open('results/model_outputs/advanced_gbm_summary.json', 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(summary), f, indent=2, ensure_ascii=False)

def generate_jump_sample_data(start_price=50000, days=180):
    """
    Generate sample data with realistic jump characteristics
    """
    np.random.seed(42)
    
    periods = days * 6  # 4-hour intervals
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='4H')
    
    # Merton model parameters
    mu = 0.6      # 60% annual drift
    sigma = 0.8   # 80% annual volatility  
    lambda_jump = 12  # 12 jumps per year
    mu_jump = -0.02   # Slightly negative jumps on average
    sigma_jump = 0.08  # 8% jump volatility
    
    dt = 1/365/6  # 4-hour intervals
    n = len(dates)
    
    # Generate price path with jumps
    prices = np.zeros(n)
    prices[0] = start_price
    
    for t in range(1, n):
        # Continuous part
        dW = np.random.normal(0, np.sqrt(dt))
        continuous_return = (mu - 0.5 * sigma**2) * dt + sigma * dW
        
        # Jump part
        jump_occurred = np.random.poisson(lambda_jump * dt)
        total_jump = 0
        
        if jump_occurred > 0:
            for _ in range(jump_occurred):
                jump_size = np.random.normal(mu_jump, sigma_jump)
                total_jump += jump_size
        
        # Update price
        prices[t] = prices[t-1] * np.exp(continuous_return + total_jump)
    
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
    
    # Run advanced analysis
    try:
        gbm_model, merton_model, jump_results = main()
        print(f"\nAdvanced GBM Analysis Successfully Completed!")
        print(f"Jumps detected: {jump_results['n_jumps']}")
        print(f"Jump frequency: {jump_results['jump_frequency']*100:.2f}%")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
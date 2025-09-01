#!/usr/bin/env python3
"""
Heston 隨機波動率模型演示腳本

這個腳本演示 Heston 模型的完整功能：
1. 數據載入和預處理
2. 參數估計和校正
3. Monte Carlo 模擬
4. 價格預測和風險分析
5. 波動率動態分析
6. 與基礎 GBM 模型比較

Usage: python scripts/demo_heston.py

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
from models.foundations.gbm_heston import HestonModel
from models.foundations.gbm_base import GBMModel
from data_collection.binance_api import BinanceDataCollector

def load_sample_data():
    """載入示例數據"""
    print("Loading BTC price data...")
    
    collector = BinanceDataCollector(verify_ssl=False)
    
    try:
        # 獲取最近 6 個月的 4 小時數據
        btc_data = collector.get_multiple_periods_data(
            symbol='BTCUSDT',
            interval='4h',
            days=180,
            market_type='spot'
        )
        
        if btc_data.empty:
            print("   Failed to get BTC data, using simulated data")
            btc_data = generate_sample_data()
        else:
            print(f"   Successfully loaded {len(btc_data)} data points")
            print(f"   Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
            print(f"   Price range: ${btc_data['close'].min():,.2f} - ${btc_data['close'].max():,.2f}")
            
            # Save data
            collector.save_data_to_csv(btc_data, 'BTC_180days_4h_heston.csv')
            
        return btc_data
        
    except Exception as e:
        print(f"   API request failed: {e}")
        print("   Using simulated data...")
        return generate_sample_data()

def generate_sample_data(start_price=50000, days=180):
    """
    Generate sample BTC data with realistic characteristics for Heston model
    """
    np.random.seed(42)
    
    periods = days * 6  # 4-hour intervals
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='4H')
    
    # Simulate Heston-like process with stochastic volatility
    mu = 0.0002  # Per period drift
    kappa = 2.0  # Mean reversion speed
    theta = 0.04  # Long-term volatility
    sigma_v = 0.3  # Vol of vol
    rho = -0.7  # Correlation
    v0 = 0.04  # Initial volatility
    dt = 1/(365*6)  # 4-hour intervals in years
    
    # Initialize arrays
    n = len(dates)
    prices = np.zeros(n)
    volatilities = np.zeros(n)
    
    prices[0] = start_price
    volatilities[0] = v0
    
    # Generate correlated random numbers
    for t in range(1, n):
        z1 = np.random.normal()
        z2_indep = np.random.normal()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * z2_indep
        
        # Update volatility (prevent negative values)
        v_curr = max(volatilities[t-1], 1e-8)
        volatilities[t] = max(
            v_curr + kappa * (theta - v_curr) * dt + sigma_v * np.sqrt(v_curr * dt) * z2,
            1e-8
        )
        
        # Update price
        prices[t] = prices[t-1] * np.exp(
            (mu - 0.5 * v_curr) * dt + np.sqrt(v_curr * dt) * z1
        )
    
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
    
    print(f"   Generated {len(data)} synthetic Heston-like data points")
    return data

def compare_with_basic_gbm(heston_model, data):
    """與基礎 GBM 模型比較"""
    print("\nComparing Heston vs Basic GBM...")
    
    # 創建基礎 GBM 模型
    gbm_model = GBMModel(data, freq='4H')
    gbm_params = gbm_model.estimate_parameters()
    
    # 模擬比較
    n_paths = 1000
    n_steps = 126  # 半年
    
    heston_sim = heston_model.simulate_paths(n_paths=n_paths, n_steps=n_steps)
    gbm_sim = gbm_model.simulate_paths(n_paths=n_paths, n_steps=n_steps)
    
    # 風險度量比較
    heston_var = heston_model.calculate_var_cvar(time_horizon=7)
    gbm_var = gbm_model.calculate_var_cvar(time_horizon=7)
    
    # 創建比較圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 最終價格分佈比較
    heston_final = heston_sim['final_prices']
    gbm_final = gbm_sim['final_prices']
    
    axes[0, 0].hist(heston_final, bins=50, alpha=0.7, density=True, 
                   label='Heston', color='blue')
    axes[0, 0].hist(gbm_final, bins=50, alpha=0.7, density=True, 
                   label='Basic GBM', color='red')
    axes[0, 0].set_title('Final Price Distribution Comparison')
    axes[0, 0].set_xlabel('Price')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 路徑比較
    sample_paths = 5
    time_index = heston_sim['prices'].index
    
    for i in range(sample_paths):
        axes[0, 1].plot(time_index, heston_sim['prices'].iloc[:, i], 
                       'b-', alpha=0.6, linewidth=1)
        axes[0, 1].plot(time_index, gbm_sim['prices'].iloc[:, i], 
                       'r-', alpha=0.6, linewidth=1)
    
    axes[0, 1].plot([], [], 'b-', label='Heston', linewidth=2)
    axes[0, 1].plot([], [], 'r-', label='Basic GBM', linewidth=2)
    axes[0, 1].set_title('Sample Price Paths Comparison')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 波動率路徑（只有 Heston）
    if 'volatilities' in heston_sim:
        for i in range(sample_paths):
            vol_path = np.sqrt(heston_sim['volatilities'].iloc[:, i]) * 100
            axes[1, 0].plot(time_index, vol_path, 'b-', alpha=0.6, linewidth=1)
        
        axes[1, 0].axhline(y=np.sqrt(heston_model.theta)*100, color='green', 
                          linestyle='--', label='Long-term Mean')
        axes[1, 0].set_title('Heston Volatility Paths')
        axes[1, 0].set_ylabel('Volatility (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. VaR 比較
    var_comparison = {
        'Heston 95% VaR': heston_var['VaR_95'],
        'GBM 95% VaR': gbm_var['VaR_95'],
        'Heston 99% VaR': heston_var['VaR_99'],
        'GBM 99% VaR': gbm_var['VaR_99']
    }
    
    bars = axes[1, 1].bar(range(len(var_comparison)), list(var_comparison.values()),
                         color=['blue', 'red', 'darkblue', 'darkred'])
    axes[1, 1].set_xticks(range(len(var_comparison)))
    axes[1, 1].set_xticklabels(var_comparison.keys(), rotation=45, ha='right')
    axes[1, 1].set_title('VaR Comparison (7-day)')
    axes[1, 1].set_ylabel('VaR ($)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加數值標籤
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'${height:,.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/plots/heston_vs_gbm_comparison.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 打印比較結果
    print("\nModel Comparison Summary:")
    print("=" * 50)
    print(f"Heston Model:")
    print(f"   Mean reversion speed (κ): {heston_model.kappa:.4f}")
    print(f"   Long-term volatility (θ): {np.sqrt(heston_model.theta)*100:.2f}%")
    print(f"   Volatility of volatility (σᵥ): {heston_model.sigma_v:.4f}")
    print(f"   Correlation (ρ): {heston_model.rho:.4f}")
    print(f"   95% VaR (7-day): ${heston_var['VaR_95']:,.0f}")
    
    print(f"\nBasic GBM Model:")
    print(f"   Annual drift: {gbm_params['mu_annual']*100:.2f}%")
    print(f"   Annual volatility: {gbm_params['sigma_annual']*100:.2f}%")
    print(f"   95% VaR (7-day): ${gbm_var['VaR_95']:,.0f}")
    
    return {'heston': heston_var, 'gbm': gbm_var}

def main():
    """主演示函數"""
    print("=== Heston Stochastic Volatility Model Demo ===")
    print("=" * 60)
    
    try:
        # 1. 載入數據
        data = load_sample_data()
        
        # 2. 創建並配置 Heston 模型
        print("\n2. Initializing Heston Model...")
        heston_model = HestonModel(data, price_column='close', freq='4H')
        
        # 3. Parameter estimation
        print("\n3. Estimating Parameters...")
        params = heston_model.estimate_parameters(method='mle', use_global_optimizer=False)  # Use local optimizer for faster execution
        
        # 4. Volatility dynamics analysis
        print("\n4. Analyzing Volatility Dynamics...")
        vol_analysis = heston_model.analyze_volatility_dynamics()
        
        # 5. Monte Carlo simulation
        print("\n5. Running Monte Carlo Simulation...")
        simulation = heston_model.simulate_paths(n_paths=1000, n_steps=126)  # Smaller simulation for demo
        
        # 6. Price forecasts
        print("\n6. Generating Price Forecasts...")
        forecast_30d = heston_model.forecast_price(time_horizon=30)
        forecast_90d = heston_model.forecast_price(time_horizon=90)
        
        # 7. Risk analysis
        print("\n7. Calculating Risk Metrics...")
        risk_metrics = heston_model.calculate_var_cvar(
            portfolio_value=100000,
            confidence_levels=[0.95, 0.99],
            time_horizon=7
        )
        
        # 8. Compare with basic GBM
        comparison = compare_with_basic_gbm(heston_model, data)
        
        # 9. Create analysis charts
        print("\n8. Creating Analysis Charts...")
        heston_model.plot_analysis()
        
        # 10. Save results summary
        print("\n9. Saving Results Summary...")
        
        summary = {
            'model': 'Heston Stochastic Volatility',
            'data_period': f"{data.index[0]} to {data.index[-1]}",
            'parameters': params,
            'volatility_analysis': vol_analysis,
            'forecasts': {
                '30_day': forecast_30d,
                '90_day': forecast_90d
            },
            'risk_metrics': risk_metrics,
            'simulation_summary': {
                'n_paths': simulation['n_paths'],
                'n_steps': simulation['n_steps'],
                'final_price_mean': float(np.mean(simulation['final_prices'])),
                'final_price_std': float(np.std(simulation['final_prices']))
            }
        }
        
        # 保存為 JSON
        import json
        os.makedirs('results/models', exist_ok=True)
        with open('results/models/heston_results.json', 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        print("\nHeston Model Analysis Complete!")
        print("=" * 60)
        print("Results saved to:")
        print("   Charts: results/plots/heston_analysis.png")
        print("   Comparison: results/plots/heston_vs_gbm_comparison.png") 
        print("   Summary: results/models/heston_results.json")
        
        # Results summary
        current_price = data['close'].iloc[-1]
        print(f"\nKey Results Summary:")
        print(f"   Current BTC Price: ${current_price:,.2f}")
        print(f"   30-day Expected Price: ${forecast_30d['expected_price']:,.2f}")
        print(f"   30-day Price Change: {((forecast_30d['expected_price']/current_price)-1)*100:+.2f}%")
        print(f"   Current Volatility: {np.sqrt(heston_model.v0)*100:.2f}%")
        print(f"   Long-term Volatility: {np.sqrt(heston_model.theta)*100:.2f}%")
        print(f"   Mean Reversion Half-life: {vol_analysis['half_life']*365:.0f} days")
        print(f"   95% VaR (7-day): ${risk_metrics['VaR_95']:,.0f}")
        print(f"   Volatility-Price Correlation: {heston_model.rho:.3f}")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
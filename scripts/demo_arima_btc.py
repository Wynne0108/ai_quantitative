"""
Demo Script: ARIMA Model Analysis for BTC Price Prediction

This script demonstrates how to:
1. Collect real BTC data from Binance API
2. Apply ARIMA model for price prediction
3. Evaluate model performance
4. Generate forecasts

Usage: python scripts/demo_arima_btc.py

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

# Import custom modules
from data_collection.binance_api import BinanceDataCollector
from models.foundations.arima_model import ARIMAModel

def main():
    """
    Main function to run ARIMA analysis on BTC data
    """
    print("=== BTC ARIMA 分析演示 ===")
    
    # Step 1: Collect BTC data
    print("\n1. 正在收集 BTC 數據...")
    collector = BinanceDataCollector()
    
    try:
        # Get 30 days of hourly BTC data
        btc_data = collector.get_multiple_periods_data(
            symbol='BTCUSDT',
            interval='1h', 
            days=30,
            market_type='spot'
        )
        
        if btc_data.empty:
            print("❌ 無法獲取 BTC 數據，使用模擬數據代替")
            btc_data = generate_sample_data()
        else:
            print(f"✅ 成功獲取 {len(btc_data)} 個數據點")
            print(f"數據期間: {btc_data.index[0]} 到 {btc_data.index[-1]}")
            
            # Save data
            collector.save_data_to_csv(btc_data, 'BTC_30days_1h.csv')
            
    except Exception as e:
        print(f"⚠️  API 請求失敗: {e}")
        print("使用模擬數據代替...")
        btc_data = generate_sample_data()
    
    # Step 2: Initialize ARIMA model
    print("\n2. 初始化 ARIMA 模型...")
    arima_model = ARIMAModel(btc_data, price_column='close')
    
    # Step 3: Check stationarity
    print("\n3. 檢測時間序列平穩性...")
    stationarity_result = arima_model.check_stationarity()
    
    if not stationarity_result['is_stationary']:
        print("序列非平穩，進行差分處理...")
        stationary_series = arima_model.make_stationary()
    
    # Step 4: Find optimal parameters
    print("\n4. 尋找最優 ARIMA 參數...")
    best_params, results_df = arima_model.find_best_arima_params(
        p_range=range(0, 3),
        d_range=range(0, 3), 
        q_range=range(0, 3)
    )
    
    # Step 5: Fit model
    print("\n5. 擬合 ARIMA 模型...")
    fitted_model = arima_model.fit_arima(order=best_params)
    
    # Step 6: Generate forecasts
    print("\n6. 生成價格預測...")
    forecast_steps = 24  # 24 hours
    forecast_result = arima_model.forecast(steps=forecast_steps)
    
    # Step 7: Display results
    print("\n7. 預測結果:")
    current_price = btc_data['close'].iloc[-1]
    predicted_price_24h = forecast_result['forecast'].iloc[-1]
    change_pct = ((predicted_price_24h - current_price) / current_price) * 100
    
    print(f"當前 BTC 價格: ${current_price:,.2f}")
    print(f"24小時後預測價格: ${predicted_price_24h:,.2f}")
    print(f"預測變化: {change_pct:+.2f}%")
    
    if change_pct > 0:
        print("📈 模型預測價格將上漲")
    else:
        print("📉 模型預測價格將下跌")
    
    # Step 8: Model evaluation
    print("\n8. 模型評估:")
    evaluation_metrics = arima_model.evaluate_model()
    
    # Step 9: Create visualizations
    print("\n9. 生成圖表...")
    
    # Plot price and forecast
    plt.figure(figsize=(15, 10))
    
    # Historical price
    plt.subplot(2, 2, 1)
    recent_data = btc_data['close'].tail(200)
    plt.plot(recent_data.index, recent_data.values, label='BTC 價格')
    plt.title('BTC 歷史價格 (最近200小時)')
    plt.ylabel('價格 (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Forecast plot
    plt.subplot(2, 2, 2)
    last_100 = btc_data['close'].tail(100)
    forecast_data = forecast_result['forecast']
    
    plt.plot(last_100.index, last_100.values, label='歷史價格', color='blue')
    plt.plot(forecast_data.index, forecast_data.values, 
             label='預測價格', color='red', linestyle='--')
    plt.fill_between(forecast_data.index, 
                     forecast_result['lower_ci'],
                     forecast_result['upper_ci'],
                     alpha=0.3, color='red', label='95% 信心區間')
    plt.title('ARIMA 價格預測')
    plt.ylabel('價格 (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Returns distribution
    plt.subplot(2, 2, 3)
    returns = btc_data['close'].pct_change().dropna()
    plt.hist(returns, bins=50, alpha=0.7, density=True)
    plt.title('BTC 收益率分佈')
    plt.xlabel('收益率')
    plt.ylabel('密度')
    plt.grid(True, alpha=0.3)
    
    # Model residuals
    plt.subplot(2, 2, 4)
    residuals = fitted_model.resid
    plt.plot(residuals.tail(200), alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('模型殘差 (最近200小時)')
    plt.xlabel('時間')
    plt.ylabel('殘差')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/btc_arima_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 10: Save results
    print("\n10. 保存結果...")
    
    # Save forecast results
    forecast_df = pd.DataFrame({
        '預測價格': forecast_result['forecast'],
        '下界_95%': forecast_result['lower_ci'], 
        '上界_95%': forecast_result['upper_ci']
    })
    
    # Create results directory if it doesn't exist
    os.makedirs('results/model_outputs', exist_ok=True)
    forecast_df.to_csv('results/model_outputs/btc_arima_forecast.csv')
    
    # Save model summary
    model_summary = {
        'model': f'ARIMA{best_params}',
        'data_period': f"{btc_data.index[0]} to {btc_data.index[-1]}",
        'data_points': len(btc_data),
        'current_price': float(current_price),
        'predicted_price_24h': float(predicted_price_24h),
        'predicted_change_pct': float(change_pct),
        'model_aic': float(fitted_model.aic),
        'model_bic': float(fitted_model.bic),
        'rmse': float(evaluation_metrics['rmse']),
        'mape': float(evaluation_metrics['mape']),
        'mae': float(evaluation_metrics['mae'])
    }
    
    import json
    with open('results/model_outputs/btc_arima_summary.json', 'w', encoding='utf-8') as f:
        json.dump(model_summary, f, indent=2, ensure_ascii=False)
    
    print("✅ 分析完成！結果已保存至 results/ 目錄")
    print(f"📊 圖表: results/plots/btc_arima_analysis.png")
    print(f"📈 預測結果: results/model_outputs/btc_arima_forecast.csv")
    print(f"📋 模型摘要: results/model_outputs/btc_arima_summary.json")
    
    return arima_model, forecast_result, model_summary

def generate_sample_data(start_price=30000, days=30):
    """
    Generate sample BTC data when API is not available
    """
    np.random.seed(42)
    
    periods = days * 24
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='H')
    
    # Simulate realistic BTC price movement
    returns = np.random.normal(0.0001, 0.015, periods)
    
    prices = [start_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    close_prices = np.array(prices[1:])
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
    os.makedirs('data/raw/binance', exist_ok=True)
    
    # Run analysis
    try:
        model, forecast, summary = main()
        print(f"\n🎉 ARIMA 分析成功完成！")
        print(f"最佳模型: {summary['model']}")
        print(f"預測準確度 (MAPE): {summary['mape']:.2f}%")
        
    except Exception as e:
        print(f"\n❌ 分析過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()
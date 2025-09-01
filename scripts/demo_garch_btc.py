"""
GARCH 模型演示腳本 - BTC 波動率預測

這個腳本展示如何使用 GARCH 模型來分析 BTC 波動率並計算風險指標

Usage: python scripts/demo_garch_btc.py

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

# Set matplotlib style for better plots
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False

# Import custom modules
from data_collection.binance_api import BinanceDataCollector
from models.foundations.garch_model import GARCHModel

def main():
    """
    Main function to run GARCH analysis on BTC data
    """
    print("=== BTC GARCH 波動率分析演示 ===")
    
    # Step 1: Collect BTC data
    print("\n1. 正在收集 BTC 數據...")
    collector = BinanceDataCollector(verify_ssl=False)
    
    try:
        # Get 60 days of 4-hour BTC data for volatility analysis
        btc_data = collector.get_multiple_periods_data(
            symbol='BTCUSDT',
            interval='4h',
            days=60,
            market_type='spot'
        )
        
        if btc_data.empty:
            print("❌ 無法獲取 BTC 數據，使用模擬數據代替")
            btc_data = generate_sample_data()
        else:
            print(f"✅ 成功獲取 {len(btc_data)} 個數據點")
            print(f"數據期間: {btc_data.index[0]} 到 {btc_data.index[-1]}")
            
            # Save data
            collector.save_data_to_csv(btc_data, 'BTC_60days_4h_garch.csv')
            
    except Exception as e:
        print(f"⚠️  API 請求失敗: {e}")
        print("使用模擬數據代替...")
        btc_data = generate_sample_data()
    
    # Step 2: Initialize GARCH model
    print("\n2. 初始化 GARCH 模型...")
    garch_model = GARCHModel(btc_data, price_column='close')
    
    # Step 3: Analyze returns characteristics
    print("\n3. 分析收益率特徵...")
    returns_stats = garch_model.analyze_returns()
    
    # Step 4: Test for ARCH effects
    print("\n4. 檢測 ARCH 效應...")
    arch_results = garch_model.test_arch_effects()
    
    if not arch_results['has_arch_effects']:
        print("⚠️  注意：未檢測到強烈的 ARCH 效應，但仍繼續 GARCH 分析")
    
    # Step 5: Fit GARCH model
    print("\n5. 擬合 GARCH 模型...")
    fitted_model = garch_model.fit_garch(p=1, q=1, vol_model='GARCH')
    
    # Step 6: Model diagnostics
    print("\n6. 模型診斷...")
    garch_model.model_diagnostics()
    
    # Step 7: Volatility forecasting
    print("\n7. 波動率預測...")
    forecast_result = garch_model.forecast_volatility(horizon=24, method='simulation')
    
    # Step 8: Risk metrics calculation
    print("\n8. 風險指標計算...")
    risk_1d = garch_model.calculate_var_cvar(confidence_levels=[0.95, 0.99], horizon=1)
    risk_7d = garch_model.calculate_var_cvar(confidence_levels=[0.95, 0.99], horizon=7)
    
    # Step 9: Create visualizations
    print("\n9. 創建分析圖表...")
    garch_model.plot_volatility_analysis()
    
    # Step 10: Generate detailed analysis
    print("\n10. 詳細分析報告...")
    current_price = btc_data['close'].iloc[-1]
    current_vol = np.sqrt(fitted_model.conditional_volatility.iloc[-1])
    avg_forecast_vol = forecast_result['volatility_forecast'].mean()
    
    print(f"\n📊 GARCH 模型分析結果:")
    print(f"   當前 BTC 價格: ${current_price:,.2f}")
    print(f"   當前波動率: {current_vol:.4f}%")
    print(f"   預測平均波動率: {avg_forecast_vol:.4f}%")
    print(f"   波動率變化: {((avg_forecast_vol/current_vol)-1)*100:+.2f}%")
    
    # Risk analysis
    var_95_1d = abs(risk_1d['VaR_95'])
    var_95_7d = abs(risk_7d['VaR_95'])
    
    print(f"\n🎯 風險分析:")
    print(f"   1日 95% VaR: {var_95_1d:.2f}%")
    print(f"   7日 95% VaR: {var_95_7d:.2f}%")
    
    # Investment implications
    portfolio_value = 100000  # $100,000 portfolio
    daily_risk = portfolio_value * var_95_1d / 100
    weekly_risk = portfolio_value * var_95_7d / 100
    
    print(f"\n💰 投資組合風險 (${portfolio_value:,} 投資):")
    print(f"   日風險: ${daily_risk:,.0f}")
    print(f"   週風險: ${weekly_risk:,.0f}")
    
    # Trading recommendations
    print(f"\n🎯 交易建議:")
    vol_percentile = (current_vol <= fitted_model.conditional_volatility).mean() * 100
    
    if vol_percentile >= 80:
        print(f"   🔴 當前波動率處於高位 ({vol_percentile:.0f}百分位)")
        print(f"   建議: 降低槓桿，增加風險控制")
    elif vol_percentile <= 20:
        print(f"   🟢 當前波動率處於低位 ({vol_percentile:.0f}百分位)")
        print(f"   建議: 可適當增加倉位，但仍需謹慎")
    else:
        print(f"   🟡 當前波動率處於正常範圍 ({vol_percentile:.0f}百分位)")
        print(f"   建議: 維持正常交易策略")
    
    # Step 11: Save results
    print("\n11. 保存結果...")
    
    # Save volatility forecast
    vol_forecast_df = pd.DataFrame({
        '預測波動率': forecast_result['volatility_forecast'],
        '預測方差': forecast_result['variance_forecast']
    })
    
    os.makedirs('results/model_outputs', exist_ok=True)
    vol_forecast_df.to_csv('results/model_outputs/btc_garch_volatility_forecast.csv')
    
    # Save model summary
    model_summary = {
        'model': 'GARCH(1,1)',
        'data_period': f"{btc_data.index[0]} to {btc_data.index[-1]}",
        'data_points': len(btc_data),
        'current_price': float(current_price),
        'current_volatility': float(current_vol),
        'forecast_avg_volatility': float(avg_forecast_vol),
        'volatility_change_pct': float(((avg_forecast_vol/current_vol)-1)*100),
        'var_95_1d': float(var_95_1d),
        'cvar_95_1d': float(risk_1d['CVaR_95']),
        'var_95_7d': float(var_95_7d),
        'cvar_95_7d': float(risk_7d['CVaR_95']),
        'model_aic': float(fitted_model.aic),
        'model_bic': float(fitted_model.bic),
        'arch_effects': bool(arch_results['has_arch_effects']),
        'volatility_percentile': float(vol_percentile)
    }
    
    import json
    with open('results/model_outputs/btc_garch_summary.json', 'w', encoding='utf-8') as f:
        json.dump(model_summary, f, indent=2, ensure_ascii=False)
    
    print("✅ GARCH 分析完成！結果已保存")
    print(f"📊 波動率預測: results/model_outputs/btc_garch_volatility_forecast.csv")
    print(f"📋 模型摘要: results/model_outputs/btc_garch_summary.json")
    
    return garch_model, forecast_result, model_summary

def generate_sample_data(start_price=50000, days=60):
    """
    Generate sample BTC data with GARCH volatility when API is not available
    """
    np.random.seed(42)
    
    periods = days * 6  # 4-hour intervals
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='4H')
    
    # Simulate GARCH process
    omega, alpha, beta = 0.1, 0.1, 0.8
    n = len(dates)
    returns = np.zeros(n)
    volatility = np.zeros(n)
    volatility[0] = 2.0  # Initial volatility 2%
    
    for t in range(1, n):
        volatility[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatility[t-1]**2)
        returns[t] = volatility[t] * np.random.normal(0, 1)
    
    # Convert to price data
    prices = start_price * np.exp(np.cumsum(returns / 100))
    
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
        garch_model, forecast, summary = main()
        print(f"\n🎉 GARCH 分析成功完成！")
        print(f"模型: {summary['model']}")
        print(f"當前波動率: {summary['current_volatility']:.4f}%")
        print(f"95% VaR (1日): {summary['var_95_1d']:.2f}%")
        
    except Exception as e:
        print(f"\n❌ 分析過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()
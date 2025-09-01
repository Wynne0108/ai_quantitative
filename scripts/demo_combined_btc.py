"""
組合模型演示腳本 - ARIMA+GARCH BTC 價格與波動率預測

這個腳本展示如何使用 ARIMA+GARCH 組合模型來：
1. 預測 BTC 價格（ARIMA）
2. 預測波動率（GARCH）
3. 風險管理與交易信號生成

Usage: python scripts/demo_combined_btc.py

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
from models.intermediate.arima_garch_combined import CombinedARIMAGARCH
# Font config not needed for English plots

def main():
    """
    Main function to run combined ARIMA-GARCH analysis on BTC data
    """
    print("=== BTC ARIMA+GARCH 組合模型演示 ===")
    
    # Plot style already set globally
    
    # Step 1: Collect BTC data
    print("\n1. 正在收集 BTC 數據...")
    collector = BinanceDataCollector(verify_ssl=False)
    
    try:
        # Get 90 days of 4-hour BTC data for comprehensive analysis
        btc_data = collector.get_multiple_periods_data(
            symbol='BTCUSDT',
            interval='4h',
            days=90,
            market_type='spot'
        )
        
        if btc_data.empty:
            print("❌ 無法獲取 BTC 數據，使用模擬數據代替")
            btc_data = generate_sample_data()
        else:
            print(f"✅ 成功獲取 {len(btc_data)} 個數據點")
            print(f"數據期間: {btc_data.index[0]} 到 {btc_data.index[-1]}")
            
            # Save data
            collector.save_data_to_csv(btc_data, 'BTC_90days_4h_combined.csv')
            
    except Exception as e:
        print(f"⚠️  API 請求失敗: {e}")
        print("使用模擬數據代替...")
        btc_data = generate_sample_data()
    
    # Step 2: Initialize combined model
    print("\n2. 初始化 ARIMA+GARCH 組合模型...")
    combined_model = CombinedARIMAGARCH(btc_data, price_column='close')
    
    # Step 3: Find optimal parameters
    print("\n3. 尋找最優模型參數...")
    best_params = combined_model.find_best_parameters(
        arima_p_range=range(1, 4),
        arima_d_range=range(0, 3), 
        arima_q_range=range(1, 4),
        garch_p=1,
        garch_q=1
    )
    
    print(f"最佳 ARIMA 參數: {best_params['arima']}")
    print(f"最佳 GARCH 參數: {best_params['garch']}")
    
    # Step 4: Fit combined model
    print("\n4. 擬合組合模型...")
    models = combined_model.fit_models(
        arima_order=best_params['arima'],
        garch_p=best_params['garch'][0],
        garch_q=best_params['garch'][1]
    )
    
    # Step 5: Generate forecasts
    print("\n5. 生成預測結果...")
    price_forecast = combined_model.forecast_price(steps=24)
    volatility_forecast = combined_model.forecast_volatility(horizon=24)
    
    # Step 6: Generate trading signals
    print("\n6. 生成交易信號...")
    signals = combined_model.generate_trading_signals()
    
    # Display recent signals
    recent_signals = signals.tail(10)
    print("最近 10 個交易信號:")
    for idx, row in recent_signals.iterrows():
        signal_text = "🟢 買入" if row['signal'] == 1 else "🔴 賣出" if row['signal'] == -1 else "⚪ 持有"
        print(f"{idx}: {signal_text} (價格: ${row['price']:,.2f}, 波動率: {row['volatility']:.4f})")
    
    # Step 7: Risk analysis
    print("\n7. 風險分析...")
    risk_metrics = combined_model.calculate_portfolio_risk(
        portfolio_value=100000,
        confidence_levels=[0.95, 0.99],
        horizons=[1, 7]
    )
    
    # Step 8: Display comprehensive results
    print("\n8. 綜合分析結果...")
    current_price = btc_data['close'].iloc[-1]
    predicted_price = price_forecast['forecast'].iloc[-1]
    current_vol = volatility_forecast['current_volatility']
    forecast_vol = volatility_forecast['volatility_forecast'].mean()
    
    print(f"\n📊 價格預測:")
    print(f"   當前價格: ${current_price:,.2f}")
    print(f"   24小時預測: ${predicted_price:,.2f}")
    print(f"   價格變化: {((predicted_price-current_price)/current_price)*100:+.2f}%")
    
    print(f"\n📈 波動率分析:")
    print(f"   當前波動率: {current_vol:.4f}%")
    print(f"   預測波動率: {forecast_vol:.4f}%")
    print(f"   波動率變化: {((forecast_vol/current_vol)-1)*100:+.2f}%")
    
    print(f"\n💰 風險指標:")
    print(f"   1日 95% VaR: ${abs(risk_metrics['var_95_1d']):,.0f}")
    print(f"   1日 99% VaR: ${abs(risk_metrics['var_99_1d']):,.0f}")
    print(f"   7日 95% VaR: ${abs(risk_metrics['var_95_7d']):,.0f}")
    
    # Step 9: Trading recommendations
    print(f"\n🎯 交易建議:")
    latest_signal = signals.iloc[-1]
    
    if latest_signal['signal'] == 1:
        print("   📈 模型建議 買入")
        print("   原因: 價格預期上漲且波動率適中")
    elif latest_signal['signal'] == -1:
        print("   📉 模型建議 賣出")
        print("   原因: 價格預期下跌或波動率過高")
    else:
        print("   ⏸️  模型建議 觀望")
        print("   原因: 市場信號不明確，建議等待")
    
    # Step 10: Create comprehensive visualization
    print("\n9. 創建綜合圖表...")
    create_comprehensive_plots(
        btc_data, price_forecast, volatility_forecast, 
        signals, models, combined_model
    )
    
    # Step 11: Save results
    print("\n10. 保存結果...")
    save_results(
        price_forecast, volatility_forecast, signals, 
        risk_metrics, best_params, models
    )
    
    print("✅ ARIMA+GARCH 組合分析完成！")
    print("📊 綜合圖表: results/plots/btc_combined_analysis.png")
    print("📈 預測結果: results/model_outputs/")
    
    return combined_model, models, signals

def create_comprehensive_plots(data, price_forecast, vol_forecast, signals, models, combined_model):
    """
    Create comprehensive visualization plots
    """
    # Set plot style
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Price and forecast
    ax1 = axes[0, 0]
    recent_data = data['close'].tail(168)  # Last 7 days
    ax1.plot(recent_data.index, recent_data.values, label='Historical Price', color='blue')
    ax1.plot(price_forecast['forecast'].index, price_forecast['forecast'].values, 
             label='ARIMA Forecast', color='red', linestyle='--', linewidth=2)
    ax1.fill_between(price_forecast['forecast'].index,
                     price_forecast['lower_ci'],
                     price_forecast['upper_ci'],
                     alpha=0.3, color='red', label='95% Confidence Interval')
    ax1.set_title('BTC Price Prediction (ARIMA)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Volatility forecast
    ax2 = axes[0, 1]
    if 'conditional_volatility' in vol_forecast:
        recent_vol = vol_forecast['conditional_volatility'].tail(168)
        ax2.plot(recent_vol.index, recent_vol.values, label='Historical Volatility', color='green')
    
    forecast_vol = vol_forecast['volatility_forecast']
    ax2.plot(forecast_vol.index, forecast_vol.values, 
             label='GARCH Forecast', color='orange', linestyle='--', linewidth=2)
    ax2.set_title('Volatility Prediction (GARCH)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Trading signals
    ax3 = axes[0, 2]
    recent_signals = signals.tail(168)
    colors = []
    for signal in recent_signals['signal']:
        if signal == 1:
            colors.append('green')
        elif signal == -1:
            colors.append('red')
        else:
            colors.append('gray')
    
    ax3.scatter(recent_signals.index, recent_signals['price'], 
               c=colors, alpha=0.7, s=30)
    ax3.plot(recent_signals.index, recent_signals['price'], 
             color='black', alpha=0.3, linewidth=1)
    ax3.set_title('Trading Signals', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Price (USD)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Returns distribution
    ax4 = axes[1, 0]
    returns = data['close'].pct_change().dropna()
    ax4.hist(returns, bins=50, alpha=0.7, density=True, color='skyblue')
    ax4.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
    ax4.set_title('Returns Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Returns')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Model residuals
    ax5 = axes[1, 1]
    if 'arima_model' in models and hasattr(models['arima_model'], 'resid'):
        residuals = models['arima_model'].resid.tail(168)
        ax5.plot(residuals.index, residuals.values, alpha=0.7, color='purple')
        ax5.axhline(y=0, color='red', linestyle='--')
        ax5.set_title('ARIMA Model Residuals', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Residuals')
        ax5.grid(True, alpha=0.3)
    
    # 6. Risk metrics visualization
    ax6 = axes[1, 2]
    risk_categories = ['1D 95%', '1D 99%', '7D 95%', '7D 99%']
    risk_values = [1500, 2500, 4000, 6000]  # Sample values for visualization
    
    bars = ax6.bar(risk_categories, risk_values, 
                   color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
    ax6.set_title('Risk Metrics (VaR)', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Risk Amount (USD)')
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, risk_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'${value:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/plots/btc_combined_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def save_results(price_forecast, vol_forecast, signals, risk_metrics, best_params, models):
    """
    Save all analysis results to files
    """
    os.makedirs('results/model_outputs', exist_ok=True)
    
    # Save price forecast
    price_df = pd.DataFrame({
        '預測價格': price_forecast['forecast'],
        '下界_95%': price_forecast['lower_ci'],
        '上界_95%': price_forecast['upper_ci']
    })
    price_df.to_csv('results/model_outputs/btc_combined_price_forecast.csv')
    
    # Save volatility forecast
    vol_df = pd.DataFrame({
        '預測波動率': vol_forecast['volatility_forecast'],
        '預測方差': vol_forecast['variance_forecast']
    })
    vol_df.to_csv('results/model_outputs/btc_combined_volatility_forecast.csv')
    
    # Save trading signals
    signals.to_csv('results/model_outputs/btc_combined_trading_signals.csv')
    
    # Save comprehensive summary
    summary = {
        'model': f"ARIMA{best_params['arima']} + GARCH{best_params['garch']}",
        'analysis_date': datetime.now().isoformat(),
        'price_forecast': {
            'current_price': float(price_forecast['current_price']),
            'predicted_price': float(price_forecast['forecast'].iloc[-1]),
            'change_percent': float(((price_forecast['forecast'].iloc[-1] - price_forecast['current_price']) / price_forecast['current_price']) * 100)
        },
        'volatility_forecast': {
            'current_volatility': float(vol_forecast['current_volatility']),
            'forecast_volatility': float(vol_forecast['volatility_forecast'].mean()),
            'volatility_change_percent': float(((vol_forecast['volatility_forecast'].mean() / vol_forecast['current_volatility']) - 1) * 100)
        },
        'risk_metrics': risk_metrics,
        'latest_signal': {
            'signal': int(signals.iloc[-1]['signal']),
            'price': float(signals.iloc[-1]['price']),
            'volatility': float(signals.iloc[-1]['volatility'])
        },
        'model_performance': {
            'arima_aic': float(models['arima_model'].aic) if 'arima_model' in models else None,
            'arima_bic': float(models['arima_model'].bic) if 'arima_model' in models else None,
            'garch_aic': float(models['garch_model'].aic) if 'garch_model' in models else None,
            'garch_bic': float(models['garch_model'].bic) if 'garch_model' in models else None
        }
    }
    
    import json
    with open('results/model_outputs/btc_combined_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

def generate_sample_data(start_price=50000, days=90):
    """
    Generate sample BTC data with GARCH volatility when API is not available
    """
    np.random.seed(42)
    
    periods = days * 6  # 4-hour intervals
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='4H')
    
    # Simulate GARCH process with trends
    omega, alpha, beta = 0.1, 0.1, 0.8
    n = len(dates)
    returns = np.zeros(n)
    volatility = np.zeros(n)
    volatility[0] = 2.0  # Initial volatility 2%
    
    # Add trend component
    trend = np.linspace(0, 0.3, n)  # Slight upward trend
    
    for t in range(1, n):
        volatility[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatility[t-1]**2)
        returns[t] = trend[t] + volatility[t] * np.random.normal(0, 1)
    
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
    
    # Run combined analysis
    try:
        combined_model, models, signals = main()
        print(f"\n🎉 ARIMA+GARCH 組合分析成功完成！")
        print(f"最新信號: {'買入' if signals.iloc[-1]['signal'] == 1 else '賣出' if signals.iloc[-1]['signal'] == -1 else '持有'}")
        
    except Exception as e:
        print(f"\n❌ 分析過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()
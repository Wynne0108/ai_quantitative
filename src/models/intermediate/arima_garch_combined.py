"""
ARIMA-GARCH 組合模型

這個模組實現了 ARIMA 和 GARCH 的組合模型，用於：
1. 同時預測價格趨勢（ARIMA）和波動率（GARCH）
2. 生成綜合交易信號
3. 計算綜合風險指標
4. 提供全面的市場分析

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import foundation models
from models.foundations.arima_model import ARIMAModel  
from models.foundations.garch_model import GARCHModel

class CombinedARIMAGARCH:
    """
    ARIMA-GARCH 組合模型類
    
    結合 ARIMA 的價格預測能力和 GARCH 的波動率建模能力，
    提供更全面的加密貨幣分析和風險管理工具
    """
    
    def __init__(self, data, price_column='close'):
        """
        初始化組合模型
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含價格數據的 DataFrame
        price_column : str
            價格欄位名稱
        """
        self.data = data
        self.price_column = price_column
        
        # Initialize individual models
        self.arima_model = ARIMAModel(data, price_column)
        self.garch_model = GARCHModel(data, price_column)
        
        # Model storage
        self.fitted_arima = None
        self.fitted_garch = None
        self.combined_forecast = None
        
        print("✅ ARIMA-GARCH 組合模型初始化完成")
        
    def find_best_parameters(self, arima_p_range=range(1, 4), arima_d_range=range(0, 3), 
                            arima_q_range=range(1, 4), garch_p=1, garch_q=1):
        """
        尋找最佳模型參數
        
        Parameters:
        -----------
        arima_p_range : range
            ARIMA p 參數搜索範圍
        arima_d_range : range  
            ARIMA d 參數搜索範圍
        arima_q_range : range
            ARIMA q 參數搜索範圍
        garch_p : int
            GARCH p 參數
        garch_q : int
            GARCH q 參數
            
        Returns:
        --------
        dict : 最佳參數組合
        """
        print("🔍 搜索最佳模型參數...")
        
        # Find best ARIMA parameters
        arima_best_params, _ = self.arima_model.find_best_arima_params(
            p_range=arima_p_range,
            d_range=arima_d_range, 
            q_range=arima_q_range
        )
        
        # GARCH parameters (usually (1,1) works well)
        garch_best_params = (garch_p, garch_q)
        
        best_params = {
            'arima': arima_best_params,
            'garch': garch_best_params
        }
        
        print(f"✅ 最佳 ARIMA 參數: {arima_best_params}")
        print(f"✅ 最佳 GARCH 參數: {garch_best_params}")
        
        return best_params
    
    def fit_models(self, arima_order=(2,1,2), garch_p=1, garch_q=1):
        """
        擬合 ARIMA 和 GARCH 模型
        
        Parameters:
        -----------
        arima_order : tuple
            ARIMA (p,d,q) 參數
        garch_p : int
            GARCH p 參數
        garch_q : int
            GARCH q 參數
            
        Returns:
        --------
        dict : 擬合的模型
        """
        print("🔧 擬合組合模型...")
        
        # Fit ARIMA model
        print("正在擬合 ARIMA 模型...")
        self.fitted_arima = self.arima_model.fit_arima(order=arima_order)
        
        # Fit GARCH model  
        print("正在擬合 GARCH 模型...")
        self.fitted_garch = self.garch_model.fit_garch(p=garch_p, q=garch_q)
        
        models = {
            'arima_model': self.fitted_arima,
            'garch_model': self.fitted_garch
        }
        
        print("✅ 組合模型擬合完成")
        return models
    
    def forecast_price(self, steps=24):
        """
        使用 ARIMA 模型預測價格
        
        Parameters:
        -----------
        steps : int
            預測步數
            
        Returns:
        --------
        dict : 價格預測結果
        """
        if self.fitted_arima is None:
            raise ValueError("ARIMA 模型必須先擬合")
            
        print(f"📈 生成 {steps} 期價格預測...")
        price_forecast = self.arima_model.forecast(steps=steps)
        price_forecast['current_price'] = self.data[self.price_column].iloc[-1]
        
        return price_forecast
    
    def forecast_volatility(self, horizon=24):
        """
        使用 GARCH 模型預測波動率
        
        Parameters:
        -----------
        horizon : int
            預測期間
            
        Returns:
        --------
        dict : 波動率預測結果
        """
        if self.fitted_garch is None:
            raise ValueError("GARCH 模型必須先擬合")
            
        print(f"📊 生成 {horizon} 期波動率預測...")
        vol_forecast = self.garch_model.forecast_volatility(horizon=horizon)
        vol_forecast['current_volatility'] = np.sqrt(self.fitted_garch.conditional_volatility.iloc[-1])
        
        return vol_forecast
    
    def generate_combined_forecast(self, price_steps=24, vol_horizon=24, confidence_level=0.95):
        """
        生成組合預測結果
        
        Parameters:
        -----------
        price_steps : int
            價格預測步數
        vol_horizon : int
            波動率預測期間
        confidence_level : float
            信心水準
            
        Returns:
        --------
        dict : 組合預測結果
        """
        if self.fitted_arima is None or self.fitted_garch is None:
            raise ValueError("模型必須先擬合才能進行組合預測")
            
        print("🎯 生成組合預測...")
        
        # Get individual forecasts
        price_forecast = self.forecast_price(steps=price_steps)
        vol_forecast = self.forecast_volatility(horizon=vol_horizon)
        
        # Calculate enhanced confidence intervals using GARCH volatility
        current_price = self.data[self.price_column].iloc[-1]
        garch_volatility = vol_forecast['volatility_forecast']
        
        # Calculate returns instead of direct predictions
        predicted_returns = (price_forecast['forecast'] / current_price - 1) * 100
        
        # Enhanced confidence intervals using GARCH volatility
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        enhanced_lower = price_forecast['forecast'] * (1 - z_score * garch_volatility / 100)
        enhanced_upper = price_forecast['forecast'] * (1 + z_score * garch_volatility / 100)
        
        # Combine results
        self.combined_forecast = {
            'price_forecast': price_forecast['forecast'],
            'volatility_forecast': vol_forecast['volatility_forecast'],
            'price_lower_ci': price_forecast['lower_ci'],
            'price_upper_ci': price_forecast['upper_ci'],
            'enhanced_lower_ci': enhanced_lower,
            'enhanced_upper_ci': enhanced_upper,
            'predicted_returns': predicted_returns,
            'confidence_level': confidence_level,
            'forecast_horizon': price_steps
        }
        
        # Summary statistics
        print(f"✅ 組合預測完成")
        print(f"當前價格: ${current_price:,.2f}")
        print(f"預測期末價格: ${price_forecast['forecast'].iloc[-1]:,.2f}")
        print(f"價格變化: {((price_forecast['forecast'].iloc[-1]/current_price)-1)*100:+.2f}%")
        print(f"平均預測波動率: {garch_volatility.mean():.4f}%")
        
        return self.combined_forecast
    
    def calculate_portfolio_risk(self, portfolio_value=100000, confidence_levels=[0.95, 0.99], horizons=[1, 7]):
        """
        計算投資組合風險指標
        
        Parameters:
        -----------
        portfolio_value : float
            投資組合價值
        confidence_levels : list
            信心水準列表
        horizons : list
            時間期間列表
            
        Returns:
        --------
        dict : 風險指標
        """
        if self.fitted_garch is None:
            raise ValueError("GARCH 模型必須先擬合才能計算風險")
            
        print("📊 計算投資組合風險...")
        
        risk_metrics = {}
        
        for horizon in horizons:
            horizon_risks = self.garch_model.calculate_var_cvar(
                confidence_levels=confidence_levels, 
                horizon=horizon
            )
            
            for conf_level in confidence_levels:
                var_key = f'VaR_{int(conf_level*100)}'
                cvar_key = f'CVaR_{int(conf_level*100)}'
                
                # Convert percentage to dollar amounts
                var_pct = abs(horizon_risks[var_key])
                var_dollar = portfolio_value * var_pct / 100
                
                cvar_pct = abs(horizon_risks[cvar_key]) 
                cvar_dollar = portfolio_value * cvar_pct / 100
                
                risk_metrics[f'var_{int(conf_level*100)}_{horizon}d'] = var_dollar
                risk_metrics[f'cvar_{int(conf_level*100)}_{horizon}d'] = cvar_dollar
        
        return risk_metrics
    
    def generate_trading_signals(self, price_threshold=0.02, volatility_threshold=0.05):
        """
        生成交易信號
        
        Parameters:
        -----------
        price_threshold : float
            價格變化閾值
        volatility_threshold : float
            波動率閾值
            
        Returns:
        --------
        pd.DataFrame : 交易信號
        """
        if self.fitted_arima is None or self.fitted_garch is None:
            raise ValueError("模型必須先擬合才能生成交易信號")
            
        print("🎯 生成交易信號...")
        
        # Get price and volatility data
        prices = self.data[self.price_column]
        returns = prices.pct_change()
        volatility = self.fitted_garch.conditional_volatility
        
        # Initialize signals
        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices
        signals['returns'] = returns
        signals['volatility'] = np.sqrt(volatility)
        signals['signal'] = 0  # 0: hold, 1: buy, -1: sell
        
        # Generate signals based on combined criteria
        for i in range(len(signals)):
            current_vol = signals['volatility'].iloc[i]
            
            if i >= 1:
                recent_return = signals['returns'].iloc[i-1:i+1].mean()
                
                # Buy signals
                if (recent_return > price_threshold and current_vol < volatility_threshold):
                    signals.loc[signals.index[i], 'signal'] = 1
                    
                # Sell signals  
                elif (recent_return < -price_threshold or current_vol > volatility_threshold):
                    signals.loc[signals.index[i], 'signal'] = -1
        
        # Add signal interpretation
        signals['signal_interpretation'] = signals['signal'].map({
            1: 'Buy', 0: 'Hold', -1: 'Sell'
        })
        
        return signals.dropna()
    
    def backtest_strategy(self, signals, initial_capital=100000, transaction_cost=0.001):
        """
        回測交易策略
        
        Parameters:
        -----------
        signals : pd.DataFrame
            交易信號數據
        initial_capital : float
            初始資金
        transaction_cost : float
            交易成本
            
        Returns:
        --------
        dict : 回測結果
        """
        print("📊 執行策略回測...")
        
        # Initialize portfolio
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['price'] = signals['price']
        portfolio['signal'] = signals['signal']
        portfolio['position'] = 0.0
        portfolio['cash'] = float(initial_capital)
        portfolio['holdings'] = 0.0
        portfolio['total'] = float(initial_capital)
        
        # Execute trades
        for i in range(len(portfolio)):
            if i == 0:
                continue
                
            current_signal = portfolio['signal'].iloc[i]
            prev_position = portfolio['position'].iloc[i-1]
            current_price = portfolio['price'].iloc[i]
            
            # Calculate position change
            if current_signal == 1 and prev_position <= 0:  # Buy signal
                # Buy with available cash
                shares_to_buy = portfolio['cash'].iloc[i-1] / current_price * (1 - transaction_cost)
                portfolio.loc[portfolio.index[i], 'position'] = shares_to_buy
                portfolio.loc[portfolio.index[i], 'cash'] = 0
                
            elif current_signal == -1 and prev_position > 0:  # Sell signal
                # Sell all positions
                cash_from_sale = prev_position * current_price * (1 - transaction_cost)
                portfolio.loc[portfolio.index[i], 'position'] = 0
                portfolio.loc[portfolio.index[i], 'cash'] = cash_from_sale
                
            else:  # Hold
                portfolio.loc[portfolio.index[i], 'position'] = prev_position
                portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
            
            # Calculate portfolio value
            portfolio.loc[portfolio.index[i], 'holdings'] = (
                portfolio['position'].iloc[i] * current_price
            )
            portfolio.loc[portfolio.index[i], 'total'] = (
                portfolio['cash'].iloc[i] + portfolio['holdings'].iloc[i]
            )
        
        # Calculate performance metrics
        final_value = portfolio['total'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Calculate benchmark (buy and hold)
        benchmark_return = (portfolio['price'].iloc[-1] - portfolio['price'].iloc[0]) / portfolio['price'].iloc[0] * 100
        
        results = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'portfolio_history': portfolio
        }
        
        print(f"初始資金: ${initial_capital:,.0f}")
        print(f"期末價值: ${final_value:,.0f}")
        print(f"總收益率: {total_return:+.2f}%")
        print(f"基準收益率: {benchmark_return:+.2f}%")
        print(f"超額收益: {total_return - benchmark_return:+.2f}%")
        
        return results


def demo_combined_analysis():
    """
    組合模型演示函數
    """
    # This would typically be called from the main demo script
    print("🎯 組合模型演示")
    print("請使用 scripts/demo_combined_btc.py 來運行完整演示")

if __name__ == "__main__":
    demo_combined_analysis()
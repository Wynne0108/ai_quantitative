"""
基礎幾何布朗運動 (Geometric Brownian Motion) 模型

這個模組實現經典的 GBM 模型，用於：
1. 資產價格建模和預測
2. Monte Carlo 模擬
3. 風險管理和 VaR 計算
4. 投資組合分析

數學模型：dS = μS dt + σS dW
解析解：S(t) = S(0) * exp((μ - σ²/2) * t + σ * W(t))

Author: Quantitative Trading Team  
Date: 2025-01-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class GBMModel:
    """
    基礎幾何布朗運動模型類
    
    實現經典 GBM 模型的所有核心功能，包括參數估計、
    模擬、預測和風險分析
    """
    
    def __init__(self, data, price_column='close', freq='4H'):
        """
        初始化 GBM 模型
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含價格數據的 DataFrame
        price_column : str
            價格欄位名稱
        freq : str
            數據頻率 ('1H', '4H', '1D' 等)
        """
        self.data = data
        self.price_column = price_column
        self.freq = freq
        
        # 計算收益率
        self.prices = data[price_column].dropna()
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        
        # 時間間隔（年化）
        self.dt = self._get_time_delta()
        
        # 模型參數
        self.mu = None      # 漂移率 (drift)
        self.sigma = None   # 波動率 (volatility)
        self.fitted = False
        
        # 模擬結果
        self.simulation_results = None
        self.forecast_results = None
        
        print(f"GBM Model Initialized Successfully")
        print(f"   Data Points: {len(self.prices)}")
        print(f"   Data Frequency: {freq}")
        print(f"   Time Delta: {self.dt:.6f} years")
    
    def _get_time_delta(self):
        """計算時間間隔（年化）"""
        freq_mapping = {
            '1min': 1/(365*24*60), '5min': 5/(365*24*60), '15min': 15/(365*24*60),
            '30min': 30/(365*24*60), '1H': 1/(365*24), '4H': 4/(365*24),
            '1D': 1/365, '1W': 7/365, '1M': 30/365
        }
        return freq_mapping.get(self.freq, 1/365)  # 預設日線
    
    def estimate_parameters(self, method='mle'):
        """
        估計 GBM 參數
        
        Parameters:
        -----------
        method : str
            參數估計方法 ('mle', 'moments')
            
        Returns:
        --------
        dict : 估計的參數
        """
        print("Estimating GBM Parameters...")
        
        if method == 'mle':
            # 最大似然估計
            self.mu, self.sigma = self._mle_estimation()
        elif method == 'moments':
            # 矩方法估計
            self.mu, self.sigma = self._moment_estimation()
        else:
            raise ValueError("method 必須是 'mle' 或 'moments'")
        
        self.fitted = True
        
        # 年化參數
        mu_annual = self.mu / self.dt
        sigma_annual = self.sigma / np.sqrt(self.dt)
        
        params = {
            'mu_per_period': self.mu,
            'sigma_per_period': self.sigma,
            'mu_annual': mu_annual,
            'sigma_annual': sigma_annual,
            'estimation_method': method
        }
        
        print(f"Parameter Estimation Complete ({method.upper()})")
        print(f"   Annual Drift Rate (mu): {mu_annual:.4f} ({mu_annual*100:.2f}%)")
        print(f"   Annual Volatility (sigma): {sigma_annual:.4f} ({sigma_annual*100:.2f}%)")
        print(f"   Per-period Drift: {self.mu:.6f}")
        print(f"   Per-period Volatility: {self.sigma:.6f}")
        
        return params
    
    def _mle_estimation(self):
        """最大似然估計"""
        n = len(self.returns)
        
        # 對數似然函數
        def neg_log_likelihood(params):
            mu, sigma = params
            if sigma <= 0:
                return np.inf
            
            # 計算對數似然
            residuals = self.returns - mu
            ll = -0.5 * n * np.log(2 * np.pi) - n * np.log(sigma) - \
                 0.5 * np.sum(residuals**2) / (sigma**2)
            return -ll
        
        # 初始猜測
        initial_mu = self.returns.mean()
        initial_sigma = self.returns.std()
        
        # 優化
        result = minimize(
            neg_log_likelihood,
            [initial_mu, initial_sigma],
            method='L-BFGS-B',
            bounds=[(-1, 1), (1e-6, 2)]
        )
        
        if result.success:
            return result.x[0], result.x[1]
        else:
            print("Warning: MLE optimization failed, using moment method")
            return self._moment_estimation()
    
    def _moment_estimation(self):
        """矩方法估計"""
        mu = self.returns.mean()
        sigma = self.returns.std()
        return mu, sigma
    
    def simulate_paths(self, n_paths=1000, n_steps=252, start_price=None, random_seed=42):
        """
        Monte Carlo 模擬價格路徑
        
        Parameters:
        -----------
        n_paths : int
            模擬路徑數量
        n_steps : int
            每條路徑的步數
        start_price : float
            起始價格，預設為最新價格
        random_seed : int
            隨機種子
            
        Returns:
        --------
        dict : 模擬結果
        """
        if not self.fitted:
            raise ValueError("必須先調用 estimate_parameters() 估計參數")
        
        print(f"Running Monte Carlo simulation ({n_paths} paths, {n_steps} steps)...")
        
        if start_price is None:
            start_price = self.prices.iloc[-1]
        
        # 設置隨機種子
        np.random.seed(random_seed)
        
        # 生成隨機數
        dt_sim = 1/252  # 日交易日
        random_shocks = np.random.normal(0, 1, (n_paths, n_steps))
        
        # 初始化價格矩陣
        prices = np.zeros((n_paths, n_steps + 1))
        prices[:, 0] = start_price
        
        # 模擬價格路徑
        for t in range(n_steps):
            prices[:, t+1] = prices[:, t] * np.exp(
                (self.mu/self.dt - 0.5 * (self.sigma/np.sqrt(self.dt))**2) * dt_sim +
                (self.sigma/np.sqrt(self.dt)) * np.sqrt(dt_sim) * random_shocks[:, t]
            )
        
        # 創建時間索引
        time_index = pd.date_range(
            start=self.prices.index[-1],
            periods=n_steps + 1,
            freq='D'
        )
        
        # 創建結果 DataFrame
        price_df = pd.DataFrame(prices.T, index=time_index)
        
        self.simulation_results = {
            'prices': price_df,
            'paths': prices,
            'start_price': start_price,
            'n_paths': n_paths,
            'n_steps': n_steps,
            'dt': dt_sim,
            'final_prices': prices[:, -1]
        }
        
        # 統計摘要
        final_prices = prices[:, -1]
        returns_sim = (final_prices - start_price) / start_price
        
        print("Simulation complete")
        print(f"   Starting price: ${start_price:,.2f}")
        print(f"   Final price statistics:")
        print(f"     Mean: ${np.mean(final_prices):,.2f}")
        print(f"     Median: ${np.median(final_prices):,.2f}")
        print(f"     Std dev: ${np.std(final_prices):,.2f}")
        print(f"   Expected return: {np.mean(returns_sim)*100:.2f}%")
        print(f"   Return volatility: {np.std(returns_sim)*100:.2f}%")
        
        return self.simulation_results
    
    def forecast_price(self, time_horizon=30, confidence_levels=[0.95]):
        """
        價格預測
        
        Parameters:
        -----------
        time_horizon : int
            預測天數
        confidence_levels : list
            信心水準列表
            
        Returns:
        --------
        dict : 預測結果
        """
        if not self.fitted:
            raise ValueError("必須先調用 estimate_parameters() 估計參數")
        
        print(f"Generating {time_horizon}-day price forecast...")
        
        current_price = self.prices.iloc[-1]
        t = time_horizon / 365  # 轉換為年
        
        # 年化參數
        mu_annual = self.mu / self.dt
        sigma_annual = self.sigma / np.sqrt(self.dt)
        
        # 預期價格（幾何平均）
        expected_price = current_price * np.exp(mu_annual * t)
        
        # 價格分佈參數
        ln_S0 = np.log(current_price)
        mean_ln_S = ln_S0 + (mu_annual - 0.5 * sigma_annual**2) * t
        std_ln_S = sigma_annual * np.sqrt(t)
        
        # 計算信心區間
        forecast_data = {
            'current_price': current_price,
            'time_horizon_days': time_horizon,
            'expected_price': expected_price,
            'mean_log_price': mean_ln_S,
            'std_log_price': std_ln_S
        }
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            z_lower = stats.norm.ppf(alpha/2)
            z_upper = stats.norm.ppf(1 - alpha/2)
            
            lower_bound = np.exp(mean_ln_S + z_lower * std_ln_S)
            upper_bound = np.exp(mean_ln_S + z_upper * std_ln_S)
            
            forecast_data[f'lower_{int(conf_level*100)}'] = lower_bound
            forecast_data[f'upper_{int(conf_level*100)}'] = upper_bound
        
        self.forecast_results = forecast_data
        
        print("Forecast complete")
        print(f"   Current price: ${current_price:,.2f}")
        print(f"   Expected price ({time_horizon} days): ${expected_price:,.2f}")
        print(f"   Expected change: {((expected_price/current_price)-1)*100:+.2f}%")
        
        for conf_level in confidence_levels:
            lower = forecast_data[f'lower_{int(conf_level*100)}']
            upper = forecast_data[f'upper_{int(conf_level*100)}']
            print(f"   {int(conf_level*100)}% confidence interval: ${lower:,.2f} - ${upper:,.2f}")
        
        return forecast_data
    
    def calculate_var_cvar(self, portfolio_value=100000, confidence_levels=[0.95, 0.99], 
                          time_horizon=1):
        """
        計算 VaR 和 CVaR
        
        Parameters:
        -----------
        portfolio_value : float
            投資組合價值
        confidence_levels : list
            信心水準列表
        time_horizon : int
            時間範圍（天）
            
        Returns:
        --------
        dict : 風險指標
        """
        if not self.fitted:
            raise ValueError("必須先調用 estimate_parameters() 估計參數")
        
        print(f"Calculating risk metrics (portfolio: ${portfolio_value:,})...")
        
        # 年化參數
        mu_annual = self.mu / self.dt  
        sigma_annual = self.sigma / np.sqrt(self.dt)
        
        # 時間調整
        t = time_horizon / 365
        mu_t = mu_annual * t
        sigma_t = sigma_annual * np.sqrt(t)
        
        risk_metrics = {
            'portfolio_value': portfolio_value,
            'time_horizon': time_horizon,
            'mu_period': mu_t,
            'sigma_period': sigma_t
        }
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            
            # VaR (基於對數正態分佈)
            z_alpha = stats.norm.ppf(alpha)
            var_return = np.exp(mu_t + z_alpha * sigma_t) - 1
            var_dollar = abs(portfolio_value * var_return)
            
            # CVaR (條件期望)
            # E[R | R < VaR] for log-normal distribution
            numerator = stats.norm.cdf(z_alpha - sigma_t)
            denominator = stats.norm.cdf(z_alpha)
            
            if denominator > 1e-10:  # 避免除零
                cvar_return = np.exp(mu_t + 0.5 * sigma_t**2) * numerator / denominator - 1
                cvar_dollar = abs(portfolio_value * cvar_return)
            else:
                cvar_dollar = var_dollar * 1.5  # 近似值
            
            risk_metrics[f'VaR_{int(conf_level*100)}'] = var_dollar
            risk_metrics[f'CVaR_{int(conf_level*100)}'] = cvar_dollar
            
            print(f"   {int(conf_level*100)}% VaR ({time_horizon}d): ${var_dollar:,.0f}")
            print(f"   {int(conf_level*100)}% CVaR ({time_horizon}d): ${cvar_dollar:,.0f}")
        
        return risk_metrics
    
    def analyze_model_fit(self):
        """
        分析模型擬合品質
        
        Returns:
        --------
        dict : 擬合品質指標
        """
        if not self.fitted:
            raise ValueError("必須先調用 estimate_parameters() 估計參數")
        
        print("Analyzing model fit quality...")
        
        # 理論 vs 實際統計
        theoretical_mean = self.mu
        theoretical_std = self.sigma
        actual_mean = self.returns.mean()
        actual_std = self.returns.std()
        
        # Jarque-Bera 正態性檢驗
        jb_stat, jb_pvalue = stats.jarque_bera(self.returns)
        
        # Kolmogorov-Smirnov 檢驗
        ks_stat, ks_pvalue = stats.kstest(
            self.returns, 
            lambda x: stats.norm.cdf(x, self.mu, self.sigma)
        )
        
        # 自相關檢驗 (Ljung-Box)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(self.returns, lags=10, return_df=True)
        lb_significant = (lb_result['lb_pvalue'] < 0.05).sum()
        
        analysis_results = {
            'theoretical_mean': theoretical_mean,
            'actual_mean': actual_mean,
            'mean_error': abs(theoretical_mean - actual_mean),
            'theoretical_std': theoretical_std,
            'actual_std': actual_std,
            'std_error': abs(theoretical_std - actual_std),
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'normality_test_pass': jb_pvalue > 0.05,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pvalue,
            'distribution_test_pass': ks_pvalue > 0.05,
            'ljung_box_significant_lags': lb_significant,
            'independence_test_pass': lb_significant < 3
        }
        
        print("Model fit analysis complete")
        print(f"   Theoretical mean: {theoretical_mean:.6f}, Actual: {actual_mean:.6f}")
        print(f"   Theoretical std: {theoretical_std:.6f}, Actual: {actual_std:.6f}")
        print(f"   Normality test: {'PASS' if jb_pvalue > 0.05 else 'FAIL'} (p={jb_pvalue:.4f})")
        print(f"   Distribution fit: {'PASS' if ks_pvalue > 0.05 else 'FAIL'} (p={ks_pvalue:.4f})")
        print(f"   Independence test: {'PASS' if lb_significant < 3 else 'FAIL'} ({lb_significant}/10 lags significant)")
        
        return analysis_results
    
    def plot_analysis(self, figsize=(15, 12)):
        """
        創建綜合分析圖表
        
        Parameters:
        -----------
        figsize : tuple
            圖表大小
        """
        if not self.fitted:
            raise ValueError("必須先調用 estimate_parameters() 估計參數")
        
        # Set plot style for English labels
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 1. 價格時間序列
        axes[0, 0].plot(self.prices.index, self.prices.values, linewidth=1, alpha=0.8)
        axes[0, 0].set_title('Price Time Series')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 收益率分佈 vs 理論
        axes[0, 1].hist(self.returns, bins=50, density=True, alpha=0.7, 
                       color='skyblue', label='Actual Returns')
        
        # 理論正態分佈
        x = np.linspace(self.returns.min(), self.returns.max(), 100)
        theoretical_pdf = stats.norm.pdf(x, self.mu, self.sigma)
        axes[0, 1].plot(x, theoretical_pdf, 'r-', linewidth=2, label='Theoretical Normal')
        
        axes[0, 1].set_title('Returns Distribution vs Theoretical')
        axes[0, 1].set_xlabel('Returns')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q 圖
        stats.probplot(self.returns, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        
        # 4. 模擬路徑（如果有）
        if self.simulation_results is not None:
            # 顯示部分路徑
            paths_to_show = min(50, self.simulation_results['n_paths'])
            sample_indices = np.random.choice(
                self.simulation_results['n_paths'], 
                paths_to_show, 
                replace=False
            )
            
            for idx in sample_indices:
                axes[1, 1].plot(
                    self.simulation_results['prices'].index,
                    self.simulation_results['prices'].iloc[:, idx],
                    alpha=0.3, linewidth=0.5, color='blue'
                )
            
            # 平均路徑
            mean_path = self.simulation_results['prices'].mean(axis=1)
            axes[1, 1].plot(mean_path.index, mean_path.values, 
                          'r-', linewidth=2, label='Mean Path')
            
            axes[1, 1].set_title(f'Simulated Price Paths ({paths_to_show} samples)')
            axes[1, 1].set_ylabel('Price')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No simulation results available\nRun simulate_paths() first', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Simulated Paths')
        
        # 5. 收益率自相關
        from statsmodels.tsa.stattools import acf
        lags = range(0, min(21, len(self.returns)//4))
        autocorr = acf(self.returns, nlags=len(lags)-1, fft=True)
        
        axes[2, 0].bar(lags, autocorr, alpha=0.7)
        axes[2, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[2, 0].axhline(y=1.96/np.sqrt(len(self.returns)), color='red', 
                          linestyle='--', alpha=0.5, label='95% Confidence')
        axes[2, 0].axhline(y=-1.96/np.sqrt(len(self.returns)), color='red', 
                          linestyle='--', alpha=0.5)
        axes[2, 0].set_title('Returns Autocorrelation')
        axes[2, 0].set_xlabel('Lags')
        axes[2, 0].set_ylabel('Autocorrelation')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 參數敏感性分析（示例）
        if self.forecast_results is not None:
            # 顯示預測區間
            mu_annual = self.mu / self.dt
            sigma_annual = self.sigma / np.sqrt(self.dt)
            
            time_range = np.linspace(1, 365, 100)
            current_price = self.prices.iloc[-1]
            
            expected_prices = current_price * np.exp(mu_annual * time_range/365)
            
            # 95% 信心區間
            t_array = time_range / 365
            std_array = sigma_annual * np.sqrt(t_array)
            mean_array = np.log(current_price) + (mu_annual - 0.5*sigma_annual**2) * t_array
            
            upper_95 = np.exp(mean_array + 1.96 * std_array)
            lower_95 = np.exp(mean_array - 1.96 * std_array)
            
            axes[2, 1].plot(time_range, expected_prices, 'b-', linewidth=2, label='Expected Price')
            axes[2, 1].fill_between(time_range, lower_95, upper_95, 
                                   alpha=0.3, color='blue', label='95% Confidence Band')
            axes[2, 1].axhline(y=current_price, color='red', linestyle='--', 
                              alpha=0.7, label='Current Price')
            
            axes[2, 1].set_title('Price Forecast (1 Year)')
            axes[2, 1].set_xlabel('Days')
            axes[2, 1].set_ylabel('Price')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'No forecast results available\nRun forecast_price() first', 
                           ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Price Forecast')
        
        plt.tight_layout()
        
        # 保存圖表
        import os
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig('results/plots/gbm_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

def demo_gbm_basic():
    """
    基礎 GBM 模型演示函數
    """
    print("🎯 基礎 GBM 模型演示")
    print("請使用 scripts/demo_gbm_basic.py 來運行完整演示")

if __name__ == "__main__":
    demo_gbm_basic()
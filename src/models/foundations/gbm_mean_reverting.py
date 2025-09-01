"""
均值回歸幾何布朗運動模型 (Mean-Reverting GBM / Ornstein-Uhlenbeck Process)

這個模組實現均值回歸 GBM 模型，用於：
1. 捕捉價格的長期均值回歸特性
2. 模擬商品價格和匯率等具有均值回歸特徵的資產
3. 識別超買/超賣區域和價格目標
4. 改善長期預測的合理性

數學模型：
dS = α(μ - ln S) S dt + σ S dW
或等價的對數價格過程：
d(ln S) = α(μ - ln S) dt + σ dW

這是一個 Ornstein-Uhlenbeck 過程的指數變換

Author: Quantitative Trading Team  
Date: 2025-01-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

class MeanRevertingGBMModel:
    """
    均值回歸幾何布朗運動模型類
    
    實現完整的均值回歸 GBM 模型，包括參數估計、
    模擬、預測和均值回歸分析
    """
    
    def __init__(self, data, price_column='close', freq='4H'):
        """
        初始化均值回歸 GBM 模型
        
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
        
        # 計算價格和收益率
        self.prices = data[price_column].dropna()
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        self.log_prices = np.log(self.prices)
        
        # 時間間隔（年化）
        self.dt = self._get_time_delta()
        
        # 模型參數
        self.alpha = None    # 均值回歸速度 (mean reversion speed)
        self.mu = None       # 長期均值（對數價格）(long-term mean)
        self.sigma = None    # 波動率 (volatility)
        self.fitted = False
        
        # 模擬結果
        self.simulation_results = None
        self.forecast_results = None
        
        print(f"Mean-Reverting GBM Model Initialized Successfully")
        print(f"   Data Points: {len(self.prices)}")
        print(f"   Data Frequency: {freq}")
        print(f"   Time Delta: {self.dt:.6f} years")
        print(f"   Price Range: ${self.prices.min():,.2f} - ${self.prices.max():,.2f}")
    
    def _get_time_delta(self):
        """計算時間間隔（年化）"""
        freq_mapping = {
            '1min': 1/(365*24*60), '5min': 5/(365*24*60), '15min': 15/(365*24*60),
            '30min': 30/(365*24*60), '1H': 1/(365*24), '4H': 4/(365*24),
            '1D': 1/365, '1W': 7/365, '1M': 30/365
        }
        return freq_mapping.get(self.freq, 1/365)
    
    def estimate_parameters(self, method='mle'):
        """
        估計均值回歸 GBM 參數
        
        Parameters:
        -----------
        method : str
            參數估計方法 ('mle', 'ols', 'moments')
            
        Returns:
        --------
        dict : 估計的參數
        """
        print("Estimating Mean-Reverting GBM Parameters...")
        
        if method == 'mle':
            self.alpha, self.mu, self.sigma = self._mle_estimation()
        elif method == 'ols':
            self.alpha, self.mu, self.sigma = self._ols_estimation()
        elif method == 'moments':
            self.alpha, self.mu, self.sigma = self._moment_estimation()
        else:
            raise ValueError("method 必須是 'mle', 'ols', 或 'moments'")
        
        self.fitted = True
        
        # 計算衍生統計量
        half_life = np.log(2) / self.alpha if self.alpha > 0 else np.inf
        long_term_price = np.exp(self.mu)
        current_price = self.prices.iloc[-1]
        
        # 計算均值回歸程度
        distance_from_mean = abs(np.log(current_price) - self.mu)
        reversion_signal = "ABOVE MEAN" if np.log(current_price) > self.mu else "BELOW MEAN"
        
        params = {
            'alpha': self.alpha,
            'mu_log': self.mu,
            'sigma': self.sigma,
            'alpha_annual': self.alpha / self.dt,
            'sigma_annual': self.sigma / np.sqrt(self.dt),
            'half_life_periods': half_life,
            'half_life_days': half_life * self.dt * 365,
            'long_term_price': long_term_price,
            'current_price': current_price,
            'distance_from_mean': distance_from_mean,
            'reversion_signal': reversion_signal,
            'estimation_method': method
        }
        
        print(f"Parameter Estimation Complete ({method.upper()})")
        print(f"   Mean Reversion Speed (α): {self.alpha:.4f}")
        print(f"   Long-term Mean Price: ${long_term_price:,.2f}")
        print(f"   Current Price: ${current_price:,.2f}")
        print(f"   Annual Volatility (σ): {params['sigma_annual']:.4f} ({params['sigma_annual']*100:.2f}%)")
        print(f"   Half-life: {params['half_life_days']:.1f} days")
        print(f"   Price Position: {reversion_signal}")
        print(f"   Distance from Mean: {distance_from_mean:.4f} log points")
        
        return params
    
    def _mle_estimation(self):
        """最大似然估計"""
        print("Using Maximum Likelihood Estimation...")
        
        def neg_log_likelihood(params):
            alpha, mu, sigma = params
            
            if alpha <= 0 or sigma <= 0:
                return np.inf
            
            try:
                n = len(self.log_prices) - 1
                log_likelihood = 0
                
                for i in range(1, len(self.log_prices)):
                    y_prev = self.log_prices.iloc[i-1]
                    y_curr = self.log_prices.iloc[i]
                    
                    # OU 過程的條件分佈
                    mean = y_prev + alpha * (mu - y_prev) * self.dt
                    var = sigma**2 * self.dt
                    
                    log_likelihood += stats.norm.logpdf(y_curr, mean, np.sqrt(var))
                
                return -log_likelihood if np.isfinite(log_likelihood) else np.inf
                
            except:
                return np.inf
        
        # 初始猜測
        mu_init = self.log_prices.mean()
        alpha_init = 1.0
        sigma_init = self.returns.std()
        
        # 參數邊界
        bounds = [
            (0.01, 10),     # alpha
            (-10, 10),      # mu
            (0.001, 2)      # sigma
        ]
        
        # 優化
        result = minimize(
            neg_log_likelihood,
            [alpha_init, mu_init, sigma_init],
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if result.success:
            return result.x
        else:
            print("Warning: MLE optimization failed, using OLS method")
            return self._ols_estimation()
    
    def _ols_estimation(self):
        """普通最小二乘法估計（離散化 OU 過程）"""
        print("Using Ordinary Least Squares Estimation...")
        
        # 準備回歸數據
        y = self.log_prices.diff().dropna()  # Δ ln(S)
        x_lag = self.log_prices.shift(1).dropna().reindex(y.index)  # ln(S_{t-1})
        
        # 構建回歸矩陣: Δ ln(S) = a + b * ln(S_{t-1}) + ε
        # 其中 a = α*μ*dt, b = -α*dt
        X = np.column_stack([np.ones(len(x_lag)), x_lag])
        
        # OLS 估計
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        a, b = coeffs
        
        # 轉換為原始參數
        alpha = -b / self.dt
        mu = -a / b if b != 0 else self.log_prices.mean()
        
        # 計算殘差標準差
        y_pred = X @ coeffs
        residuals = y - y_pred
        sigma = np.sqrt(np.mean(residuals**2) / self.dt)
        
        # 參數合理性檢查
        alpha = max(alpha, 0.01)  # 確保正的均值回歸
        sigma = max(sigma, 0.001)  # 確保正的波動率
        
        return alpha, mu, sigma
    
    def _moment_estimation(self):
        """矩方法估計"""
        print("Using Method of Moments Estimation...")
        
        # 基本統計量
        log_mean = self.log_prices.mean()
        log_var = self.log_prices.var()
        
        # 自相關估計 alpha
        log_autocorr = self.log_prices.autocorr(lag=1)
        if not np.isnan(log_autocorr) and log_autocorr < 1:
            alpha = -np.log(log_autocorr) / self.dt
        else:
            alpha = 1.0
        
        # 長期均值
        mu = log_mean
        
        # 波動率（基於創新方差）
        returns_var = self.returns.var()
        sigma = np.sqrt(returns_var / self.dt)
        
        return alpha, mu, sigma
    
    def simulate_paths(self, n_paths=1000, n_steps=252, start_price=None, random_seed=42):
        """
        Monte Carlo 模擬均值回歸價格路徑
        
        Parameters:
        -----------
        n_paths : int
            模擬路徑數量
        n_steps : int
            每條路徑的步數
        start_price : float
            起始價格
        random_seed : int
            隨機種子
            
        Returns:
        --------
        dict : 模擬結果
        """
        if not self.fitted:
            raise ValueError("必須先調用 estimate_parameters() 估計參數")
        
        print(f"Running Mean-Reverting Monte Carlo simulation ({n_paths} paths, {n_steps} steps)...")
        
        if start_price is None:
            start_price = self.prices.iloc[-1]
        
        # 設置隨機種子
        np.random.seed(random_seed)
        
        # 時間間隔
        dt = 1/252
        
        # 初始化陣列
        log_prices = np.zeros((n_paths, n_steps + 1))
        prices = np.zeros((n_paths, n_steps + 1))
        
        log_prices[:, 0] = np.log(start_price)
        prices[:, 0] = start_price
        
        # 生成隨機數
        random_shocks = np.random.normal(0, 1, (n_paths, n_steps))
        
        # 模擬對數價格過程（OU 過程）
        for t in range(n_steps):
            # OU 過程更新
            log_prices[:, t+1] = (
                log_prices[:, t] + 
                self.alpha * (self.mu - log_prices[:, t]) * dt +
                self.sigma * np.sqrt(dt) * random_shocks[:, t]
            )
            
            # 轉換為價格
            prices[:, t+1] = np.exp(log_prices[:, t+1])
        
        # 創建時間索引
        time_index = pd.date_range(
            start=self.prices.index[-1],
            periods=n_steps + 1,
            freq='D'
        )
        
        # 創建結果 DataFrame
        price_df = pd.DataFrame(prices.T, index=time_index)
        log_price_df = pd.DataFrame(log_prices.T, index=time_index)
        
        # 計算均值回歸統計
        mean_reversion_stats = self._analyze_mean_reversion_paths(log_prices)
        
        self.simulation_results = {
            'prices': price_df,
            'log_prices': log_price_df,
            'price_paths': prices,
            'log_price_paths': log_prices,
            'start_price': start_price,
            'n_paths': n_paths,
            'n_steps': n_steps,
            'dt': dt,
            'final_prices': prices[:, -1],
            'final_log_prices': log_prices[:, -1],
            'mean_reversion_stats': mean_reversion_stats
        }
        
        # 統計摘要
        final_prices = prices[:, -1]
        returns_sim = (final_prices - start_price) / start_price
        
        print("Mean-Reverting simulation complete")
        print(f"   Starting price: ${start_price:,.2f}")
        print(f"   Long-term mean price: ${np.exp(self.mu):,.2f}")
        print(f"   Final price statistics:")
        print(f"     Mean: ${np.mean(final_prices):,.2f}")
        print(f"     Median: ${np.median(final_prices):,.2f}")
        print(f"     Std dev: ${np.std(final_prices):,.2f}")
        print(f"   Mean reversion tendency: {mean_reversion_stats['reversion_tendency']:.2f}")
        print(f"   Time spent near mean: {mean_reversion_stats['time_near_mean']*100:.1f}%")
        
        return self.simulation_results
    
    def _analyze_mean_reversion_paths(self, log_prices):
        """分析模擬路徑的均值回歸特性"""
        n_paths, n_steps = log_prices.shape
        
        # 計算每條路徑偏離均值的程度
        deviations = np.abs(log_prices - self.mu)
        
        # 均值回歸傾向（路徑結束時比開始時更接近均值的比例）
        initial_deviations = deviations[:, 0]
        final_deviations = deviations[:, -1]
        reversion_tendency = np.mean(final_deviations < initial_deviations)
        
        # 在均值附近的時間比例（偏差小於 1 個標準差）
        threshold = self.sigma * np.sqrt(252)  # 年化標準差
        near_mean = deviations < threshold
        time_near_mean = np.mean(near_mean)
        
        # 平均回歸時間
        mean_reversion_times = []
        for path in range(min(100, n_paths)):  # 只分析前 100 條路徑
            path_log_prices = log_prices[path, :]
            
            # 找到第一次接近均值的時間
            for t in range(1, n_steps):
                if abs(path_log_prices[t] - self.mu) < threshold:
                    mean_reversion_times.append(t)
                    break
        
        avg_reversion_time = np.mean(mean_reversion_times) if mean_reversion_times else n_steps
        
        return {
            'reversion_tendency': reversion_tendency,
            'time_near_mean': time_near_mean,
            'avg_reversion_time': avg_reversion_time,
            'threshold_used': threshold
        }
    
    def forecast_price(self, time_horizon=30, confidence_levels=[0.95]):
        """
        價格預測（基於均值回歸特性）
        
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
        
        print(f"Generating {time_horizon}-day Mean-Reverting price forecast...")
        
        current_price = self.prices.iloc[-1]
        current_log_price = np.log(current_price)
        t = time_horizon / 365
        
        # OU 過程的解析解
        # E[ln(S_t)] = ln(S_0) * exp(-αt) + μ * (1 - exp(-αt))
        # Var[ln(S_t)] = σ²/(2α) * (1 - exp(-2αt))
        
        exp_alpha_t = np.exp(-self.alpha * t)
        
        # 條件期望和方差
        expected_log_price = current_log_price * exp_alpha_t + self.mu * (1 - exp_alpha_t)
        var_log_price = (self.sigma**2) / (2 * self.alpha) * (1 - np.exp(-2 * self.alpha * t))
        std_log_price = np.sqrt(var_log_price)
        
        # 對數正態分佈的期望價格
        expected_price = np.exp(expected_log_price + 0.5 * var_log_price)
        median_price = np.exp(expected_log_price)
        
        forecast_data = {
            'current_price': current_price,
            'time_horizon_days': time_horizon,
            'expected_price': expected_price,
            'median_price': median_price,
            'expected_log_price': expected_log_price,
            'std_log_price': std_log_price,
            'long_term_price': np.exp(self.mu),
            'reversion_progress': 1 - exp_alpha_t,
            'model_type': 'Mean-Reverting GBM'
        }
        
        # 計算信心區間（基於對數價格的正態分佈）
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            z_lower = stats.norm.ppf(alpha/2)
            z_upper = stats.norm.ppf(1 - alpha/2)
            
            lower_log = expected_log_price + z_lower * std_log_price
            upper_log = expected_log_price + z_upper * std_log_price
            
            forecast_data[f'lower_{int(conf_level*100)}'] = np.exp(lower_log)
            forecast_data[f'upper_{int(conf_level*100)}'] = np.exp(upper_log)
        
        self.forecast_results = forecast_data
        
        print("Mean-Reverting forecast complete")
        print(f"   Current price: ${current_price:,.2f}")
        print(f"   Long-term mean price: ${np.exp(self.mu):,.2f}")
        print(f"   Expected price ({time_horizon} days): ${expected_price:,.2f}")
        print(f"   Median price ({time_horizon} days): ${median_price:,.2f}")
        print(f"   Reversion progress: {(1-exp_alpha_t)*100:.1f}%")
        
        for conf_level in confidence_levels:
            lower = forecast_data[f'lower_{int(conf_level*100)}']
            upper = forecast_data[f'upper_{int(conf_level*100)}']
            print(f"   {int(conf_level*100)}% confidence interval: ${lower:,.2f} - ${upper:,.2f}")
        
        return forecast_data
    
    def calculate_var_cvar(self, portfolio_value=100000, confidence_levels=[0.95, 0.99], 
                          time_horizon=1):
        """
        計算基於均值回歸的 VaR 和 CVaR
        
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
        
        print(f"Calculating Mean-Reverting risk metrics (portfolio: ${portfolio_value:,})...")
        
        current_price = self.prices.iloc[-1]
        current_log_price = np.log(current_price)
        t = time_horizon / 365
        
        # OU 過程參數
        exp_alpha_t = np.exp(-self.alpha * t)
        expected_log_price = current_log_price * exp_alpha_t + self.mu * (1 - exp_alpha_t)
        var_log_price = (self.sigma**2) / (2 * self.alpha) * (1 - np.exp(-2 * self.alpha * t))
        std_log_price = np.sqrt(var_log_price)
        
        risk_metrics = {
            'portfolio_value': portfolio_value,
            'time_horizon': time_horizon,
            'model': 'Mean-Reverting GBM',
            'current_price': current_price,
            'expected_log_return': expected_log_price - current_log_price,
            'log_return_std': std_log_price
        }
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            
            # VaR（基於對數收益率分佈）
            z_alpha = stats.norm.ppf(alpha)
            var_log_return = expected_log_price - current_log_price + z_alpha * std_log_price
            var_dollar = portfolio_value * abs(1 - np.exp(var_log_return))
            
            # CVaR（條件期望）
            # E[L|L>VaR] for log-normal distribution
            threshold_log = expected_log_price - current_log_price + z_alpha * std_log_price
            
            # 使用截斷正態分佈計算 CVaR
            from scipy.stats import truncnorm
            a = -np.inf
            b = (threshold_log - (expected_log_price - current_log_price)) / std_log_price
            
            truncated_mean = truncnorm.mean(a, b, 
                                          loc=expected_log_price - current_log_price, 
                                          scale=std_log_price)
            
            cvar_dollar = portfolio_value * abs(1 - np.exp(truncated_mean))
            
            risk_metrics[f'VaR_{int(conf_level*100)}'] = var_dollar
            risk_metrics[f'CVaR_{int(conf_level*100)}'] = cvar_dollar
            
            print(f"   {int(conf_level*100)}% VaR ({time_horizon}日): ${var_dollar:,.0f}")
            print(f"   {int(conf_level*100)}% CVaR ({time_horizon}日): ${cvar_dollar:,.0f}")
        
        return risk_metrics
    
    def analyze_mean_reversion_strength(self):
        """
        分析均值回歸強度和特性
        
        Returns:
        --------
        dict : 均值回歸分析結果
        """
        if not self.fitted:
            raise ValueError("必須先調用 estimate_parameters() 估計參數")
        
        print("Analyzing mean reversion characteristics...")
        
        current_price = self.prices.iloc[-1]
        long_term_price = np.exp(self.mu)
        
        # 計算基本統計量
        half_life_days = np.log(2) / self.alpha * 365 / self.dt
        distance_from_mean = abs(np.log(current_price) - self.mu)
        
        # 預期回歸時間
        reversion_90_percent = -np.log(0.1) / self.alpha * 365 / self.dt  # 90% 回歸時間
        
        # 均值回歸強度分類
        if self.alpha > 2:
            strength = "Very Strong"
        elif self.alpha > 1:
            strength = "Strong"
        elif self.alpha > 0.5:
            strength = "Moderate"
        elif self.alpha > 0.1:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        # 計算價格區間
        long_term_vol = self.sigma / np.sqrt(2 * self.alpha)
        price_1std = np.exp(self.mu + long_term_vol)
        price_2std = np.exp(self.mu + 2 * long_term_vol)
        price_1std_lower = np.exp(self.mu - long_term_vol)
        price_2std_lower = np.exp(self.mu - 2 * long_term_vol)
        
        # 當前位置分析
        if np.log(current_price) > self.mu + 2 * long_term_vol:
            position = "Extremely Overvalued"
        elif np.log(current_price) > self.mu + long_term_vol:
            position = "Overvalued"
        elif np.log(current_price) > self.mu - long_term_vol:
            position = "Fair Value Range"
        elif np.log(current_price) > self.mu - 2 * long_term_vol:
            position = "Undervalued"
        else:
            position = "Extremely Undervalued"
        
        analysis_results = {
            'mean_reversion_speed': self.alpha,
            'strength_classification': strength,
            'half_life_days': half_life_days,
            'reversion_90_percent_days': reversion_90_percent,
            'current_price': current_price,
            'long_term_price': long_term_price,
            'distance_from_mean': distance_from_mean,
            'current_position': position,
            'long_term_volatility': long_term_vol,
            'price_bands': {
                'upper_2std': price_2std,
                'upper_1std': price_1std,
                'mean': long_term_price,
                'lower_1std': price_1std_lower,
                'lower_2std': price_2std_lower
            }
        }
        
        print("Mean reversion analysis complete")
        print(f"   Mean Reversion Strength: {strength}")
        print(f"   Half-life: {half_life_days:.1f} days")
        print(f"   Current Position: {position}")
        print(f"   Current Price: ${current_price:,.2f}")
        print(f"   Long-term Mean: ${long_term_price:,.2f}")
        print(f"   90% Reversion Time: {reversion_90_percent:.0f} days")
        
        return analysis_results
    
    def plot_analysis(self, figsize=(16, 12)):
        """
        創建均值回歸模型綜合分析圖表
        
        Parameters:
        -----------
        figsize : tuple
            圖表大小
        """
        if not self.fitted:
            raise ValueError("必須先調用 estimate_parameters() 估計參數")
        
        plt.rcParams['axes.unicode_minus'] = False
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 1. 價格時間序列與均值帶
        axes[0, 0].plot(self.prices.index, self.prices.values, 'b-', linewidth=1.5, alpha=0.8, label='Price')
        
        # 長期均值和信心帶
        long_term_price = np.exp(self.mu)
        long_term_vol = self.sigma / np.sqrt(2 * self.alpha)
        
        axes[0, 0].axhline(y=long_term_price, color='green', linestyle='--', linewidth=2, label='Long-term Mean')
        axes[0, 0].axhline(y=np.exp(self.mu + long_term_vol), color='orange', linestyle=':', alpha=0.7, label='+1σ Band')
        axes[0, 0].axhline(y=np.exp(self.mu - long_term_vol), color='orange', linestyle=':', alpha=0.7, label='-1σ Band')
        axes[0, 0].axhline(y=np.exp(self.mu + 2*long_term_vol), color='red', linestyle=':', alpha=0.5, label='+2σ Band')
        axes[0, 0].axhline(y=np.exp(self.mu - 2*long_term_vol), color='red', linestyle=':', alpha=0.5, label='-2σ Band')
        
        axes[0, 0].set_title('Price vs Long-term Mean with Bands')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 對數價格偏差
        log_prices = np.log(self.prices)
        deviations = log_prices - self.mu
        
        axes[0, 1].plot(deviations.index, deviations.values, 'r-', linewidth=1, alpha=0.8)
        axes[0, 1].axhline(y=0, color='green', linestyle='--', linewidth=2, label='Long-term Mean')
        axes[0, 1].axhline(y=long_term_vol, color='orange', linestyle=':', alpha=0.7, label='+1σ')
        axes[0, 1].axhline(y=-long_term_vol, color='orange', linestyle=':', alpha=0.7, label='-1σ')
        
        axes[0, 1].set_title('Log Price Deviation from Mean')
        axes[0, 1].set_ylabel('Log Price Deviation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 模擬路徑（如果有）
        if self.simulation_results is not None:
            paths_to_show = min(30, self.simulation_results['n_paths'])
            sample_indices = np.random.choice(
                self.simulation_results['n_paths'], 
                paths_to_show, 
                replace=False
            )
            
            for idx in sample_indices:
                axes[1, 0].plot(
                    self.simulation_results['prices'].index,
                    self.simulation_results['prices'].iloc[:, idx],
                    alpha=0.4, linewidth=0.5, color='blue'
                )
            
            # 均值路徑和長期均值
            mean_path = self.simulation_results['prices'].mean(axis=1)
            axes[1, 0].plot(mean_path.index, mean_path.values, 
                          'r-', linewidth=2, label='Mean Path')
            axes[1, 0].axhline(y=long_term_price, color='green', 
                              linestyle='--', linewidth=2, label='Long-term Mean')
            
            axes[1, 0].set_title(f'Simulated Mean-Reverting Paths ({paths_to_show} samples)')
            axes[1, 0].set_ylabel('Price')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No simulation results\nRun simulate_paths() first', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Simulated Mean-Reverting Paths')
        
        # 4. 最終價格分佈（如果有模擬結果）
        if self.simulation_results is not None:
            final_prices = self.simulation_results['final_prices']
            
            axes[1, 1].hist(final_prices, bins=50, density=True, alpha=0.7, 
                           color='lightblue', label='Simulated Final Prices')
            axes[1, 1].axvline(self.prices.iloc[-1], color='red', linestyle='--', 
                              linewidth=2, label='Current Price')
            axes[1, 1].axvline(long_term_price, color='green', linestyle='--', 
                              linewidth=2, label='Long-term Mean')
            
            axes[1, 1].set_title('Final Price Distribution')
            axes[1, 1].set_xlabel('Price')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No simulation results\nRun simulate_paths() first', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Final Price Distribution')
        
        # 5. 均值回歸動態
        if len(self.log_prices) > 50:
            # 計算滾動偏差
            window = 20
            rolling_deviation = log_prices.rolling(window=window).apply(
                lambda x: np.mean(np.abs(x - self.mu))
            )
            
            axes[2, 0].plot(rolling_deviation.index, rolling_deviation.values, 
                           'purple', linewidth=1.5, label=f'{window}-period Rolling Deviation')
            axes[2, 0].axhline(y=long_term_vol, color='orange', linestyle='--', 
                              alpha=0.7, label='Expected Long-term Deviation')
            
            axes[2, 0].set_title('Mean Reversion Dynamics')
            axes[2, 0].set_ylabel('Average Absolute Deviation')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 預測分析（如果有）
        if self.forecast_results is not None:
            time_horizon = self.forecast_results['time_horizon_days']
            current_price = self.forecast_results['current_price']
            
            # 顯示預測路徑
            forecast_times = np.linspace(0, time_horizon, 100)
            forecast_paths = []
            
            for t in forecast_times:
                t_years = t / 365
                exp_alpha_t = np.exp(-self.alpha * t_years)
                expected_log = np.log(current_price) * exp_alpha_t + self.mu * (1 - exp_alpha_t)
                var_log = (self.sigma**2) / (2 * self.alpha) * (1 - np.exp(-2 * self.alpha * t_years))
                std_log = np.sqrt(var_log)
                
                # 信心區間
                upper_95 = np.exp(expected_log + 1.96 * std_log)
                lower_95 = np.exp(expected_log - 1.96 * std_log)
                expected_price = np.exp(expected_log + 0.5 * var_log)
                
                forecast_paths.append([expected_price, upper_95, lower_95])
            
            forecast_paths = np.array(forecast_paths)
            
            axes[2, 1].plot(forecast_times, forecast_paths[:, 0], 'b-', linewidth=2, label='Expected Price')
            axes[2, 1].fill_between(forecast_times, forecast_paths[:, 2], forecast_paths[:, 1],
                                   alpha=0.3, color='blue', label='95% Confidence Band')
            axes[2, 1].axhline(y=current_price, color='red', linestyle='--', 
                              alpha=0.8, label='Current Price')
            axes[2, 1].axhline(y=long_term_price, color='green', linestyle='--', 
                              alpha=0.8, label='Long-term Mean')
            
            axes[2, 1].set_title(f'Mean-Reverting Price Forecast ({time_horizon} days)')
            axes[2, 1].set_xlabel('Days')
            axes[2, 1].set_ylabel('Price')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'No forecast results\nRun forecast_price() first', 
                           ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Mean-Reverting Price Forecast')
        
        plt.tight_layout()
        
        # 保存圖表
        import os
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig('results/plots/mean_reverting_gbm_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

def demo_mean_reverting_gbm():
    """
    均值回歸 GBM 模型演示函數
    """
    print("🎯 均值回歸 GBM 模型演示")
    print("請使用 scripts/demo_mean_reverting.py 來運行完整演示")

if __name__ == "__main__":
    demo_mean_reverting_gbm()
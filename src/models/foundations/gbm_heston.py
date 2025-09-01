"""
Heston 隨機波動率模型 (Heston Stochastic Volatility Model)

這個模組實現 Heston 模型，用於：
1. 捕捉波動率的時變特性和波動率聚集現象
2. 更精確的期權定價和風險管理
3. 解決固定波動率假設的限制
4. 建模波動率微笑/偏斜現象

數學模型：
dS = μS dt + √V S dW₁
dV = κ(θ - V) dt + σᵥ √V dW₂
其中 dW₁ 和 dW₂ 相關係數為 ρ

Author: Quantitative Trading Team  
Date: 2025-01-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

class HestonModel:
    """
    Heston 隨機波動率模型類
    
    實現完整的 Heston 模型，包括參數校正、模擬、
    特徵函數計算和風險分析
    """
    
    def __init__(self, data, price_column='close', freq='4H'):
        """
        初始化 Heston 模型
        
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
        
        # 時間間隔（年化）- 必須在計算實現波動率之前設置
        self.dt = self._get_time_delta()
        
        # 計算實現波動率 (使用滾動窗口)
        self.realized_vol = self._calculate_realized_volatility()
        
        # Heston 模型參數
        self.mu = None       # 漂移率 (drift)
        self.kappa = None    # 均值回歸速度 (mean reversion speed)
        self.theta = None    # 長期波動率均值 (long-term volatility mean)
        self.sigma_v = None  # 波動率的波動率 (volatility of volatility)
        self.rho = None      # 相關係數 (correlation)
        self.v0 = None       # 初始波動率 (initial volatility)
        self.fitted = False
        
        # 模擬結果
        self.simulation_results = None
        self.forecast_results = None
        
        print(f"Heston Model Initialized Successfully")
        print(f"   Data Points: {len(self.prices)}")
        print(f"   Realized Volatility Points: {len(self.realized_vol)}")
        print(f"   Data Frequency: {freq}")
        print(f"   Time Delta: {self.dt:.6f} years")
    
    def _get_time_delta(self):
        """計算時間間隔（年化）"""
        freq_mapping = {
            '1min': 1/(365*24*60), '5min': 5/(365*24*60), '15min': 15/(365*24*60),
            '30min': 30/(365*24*60), '1H': 1/(365*24), '4H': 4/(365*24),
            '1D': 1/365, '1W': 7/365, '1M': 30/365
        }
        return freq_mapping.get(self.freq, 1/365)
    
    def _calculate_realized_volatility(self, window=30):
        """
        計算實現波動率
        
        Parameters:
        -----------
        window : int
            滾動窗口大小
            
        Returns:
        --------
        pd.Series : 實現波動率序列
        """
        # 計算滾動波動率（年化）
        rolling_var = self.returns.rolling(window=window).var()
        realized_vol = np.sqrt(rolling_var / self.dt)
        
        return realized_vol.dropna()
    
    def estimate_parameters(self, method='mle', use_global_optimizer=True):
        """
        估計 Heston 參數
        
        Parameters:
        -----------
        method : str
            參數估計方法 ('mle', 'moments')
        use_global_optimizer : bool
            是否使用全局優化器
            
        Returns:
        --------
        dict : 估計的參數
        """
        print("Estimating Heston Parameters...")
        print("Note: This may take several minutes due to model complexity...")
        
        if method == 'mle':
            if use_global_optimizer:
                params = self._estimate_with_global_optimizer()
            else:
                params = self._estimate_with_local_optimizer()
        elif method == 'moments':
            params = self._estimate_with_moments()
        else:
            raise ValueError("method 必須是 'mle' 或 'moments'")
        
        # 解包參數
        self.mu, self.kappa, self.theta, self.sigma_v, self.rho, self.v0 = params
        self.fitted = True
        
        # 檢查參數合理性
        self._validate_parameters()
        
        # 年化參數
        mu_annual = self.mu / self.dt
        theta_annual = self.theta
        
        result = {
            'mu': self.mu,
            'mu_annual': mu_annual,
            'kappa': self.kappa,
            'theta': self.theta,
            'theta_annual': theta_annual,
            'sigma_v': self.sigma_v,
            'rho': self.rho,
            'v0': self.v0,
            'estimation_method': method,
            'feller_condition': 2 * self.kappa * self.theta > self.sigma_v**2
        }
        
        print(f"Parameter Estimation Complete ({method.upper()})")
        print(f"   Annual Drift Rate (μ): {mu_annual:.4f} ({mu_annual*100:.2f}%)")
        print(f"   Mean Reversion Speed (κ): {self.kappa:.4f}")
        print(f"   Long-term Vol Mean (θ): {theta_annual:.4f} ({theta_annual*100:.2f}%)")
        print(f"   Vol of Vol (σᵥ): {self.sigma_v:.4f}")
        print(f"   Correlation (ρ): {self.rho:.4f}")
        print(f"   Initial Volatility (v₀): {self.v0:.4f} ({np.sqrt(self.v0)*100:.2f}%)")
        print(f"   Feller Condition: {'Satisfied' if result['feller_condition'] else 'Violated'}")
        
        return result
    
    def _estimate_with_moments(self):
        """使用矩方法估計參數"""
        print("Using method of moments estimation...")
        
        # 基本統計量
        mu = self.returns.mean()
        
        # 波動率統計
        vol_mean = self.realized_vol.mean()
        vol_var = self.realized_vol.var()
        vol_autocorr = self.realized_vol.autocorr(lag=1)
        
        # 相關性
        aligned_returns = self.returns.reindex(self.realized_vol.index).dropna()
        aligned_vol = self.realized_vol.reindex(aligned_returns.index).dropna()
        
        if len(aligned_returns) > 0 and len(aligned_vol) > 0:
            # 計算波動率變化
            vol_changes = aligned_vol.diff().dropna()
            returns_aligned = aligned_returns.reindex(vol_changes.index).dropna()
            
            if len(returns_aligned) > 0 and len(vol_changes) > 0:
                rho = np.corrcoef(returns_aligned, vol_changes)[0, 1]
                if np.isnan(rho):
                    rho = -0.3  # 預設值
            else:
                rho = -0.3
        else:
            rho = -0.3
        
        # 參數估計
        theta = vol_mean**2
        v0 = theta
        
        # 從自相關估計 kappa
        if not np.isnan(vol_autocorr) and vol_autocorr > 0:
            kappa = -np.log(vol_autocorr) / self.dt
        else:
            kappa = 2.0  # 預設值
        
        # 從波動率方差估計 sigma_v
        if kappa > 0 and theta > 0:
            sigma_v = np.sqrt(2 * kappa * vol_var * theta) / theta
        else:
            sigma_v = 0.3
        
        # 確保參數在合理範圍內
        mu = np.clip(mu, -1, 1)
        kappa = np.clip(kappa, 0.1, 10)
        theta = np.clip(theta, 0.001, 1)
        sigma_v = np.clip(sigma_v, 0.01, 2)
        rho = np.clip(rho, -0.99, 0.99)
        v0 = np.clip(v0, 0.001, 1)
        
        return [mu, kappa, theta, sigma_v, rho, v0]
    
    def _estimate_with_local_optimizer(self):
        """使用局部優化器估計參數"""
        print("Using local optimizer (L-BFGS-B)...")
        
        # 初始猜測 (使用矩方法)
        initial_guess = self._estimate_with_moments()
        
        # 定義目標函數
        def objective(params):
            try:
                return -self._log_likelihood(params)
            except:
                return np.inf
        
        # 參數邊界
        bounds = [
            (-1, 1),        # mu
            (0.1, 10),      # kappa
            (0.001, 1),     # theta
            (0.01, 2),      # sigma_v
            (-0.99, 0.99),  # rho
            (0.001, 1)      # v0
        ]
        
        # 優化
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x
        else:
            print("Warning: Optimization failed, using method of moments")
            return initial_guess
    
    def _estimate_with_global_optimizer(self):
        """使用全局優化器估計參數"""
        print("Using global optimizer (Differential Evolution)...")
        
        # 定義目標函數
        def objective(params):
            try:
                ll = self._log_likelihood(params)
                return -ll if np.isfinite(ll) else 1e10
            except:
                return 1e10
        
        # 參數邊界
        bounds = [
            (-0.5, 0.5),    # mu
            (0.1, 5),       # kappa
            (0.01, 0.5),    # theta
            (0.1, 1),       # sigma_v
            (-0.8, 0.8),    # rho
            (0.01, 0.5)     # v0
        ]
        
        # 全局優化
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=True
        )
        
        if result.success:
            return result.x
        else:
            print("Warning: Global optimization failed, using method of moments")
            return self._estimate_with_moments()
    
    def _log_likelihood(self, params):
        """
        計算對數似然函數（簡化版本）
        
        Parameters:
        -----------
        params : list
            [mu, kappa, theta, sigma_v, rho, v0]
            
        Returns:
        --------
        float : 對數似然值
        """
        mu, kappa, theta, sigma_v, rho, v0 = params
        
        # 參數有效性檢查
        if (kappa <= 0 or theta <= 0 or sigma_v <= 0 or 
            abs(rho) >= 1 or v0 <= 0):
            return -np.inf
        
        try:
            # 使用簡化的似然函數（基於收益率的正態近似）
            n = len(self.returns)
            
            # 估計平均波動率
            avg_vol = np.sqrt(v0)
            
            # 計算似然
            log_likelihood = 0
            for r in self.returns:
                log_likelihood += stats.norm.logpdf(r, mu, avg_vol)
            
            # 加入波動率過程的懲罰項
            if len(self.realized_vol) > 1:
                vol_penalty = 0
                for i in range(1, min(len(self.realized_vol), 100)):
                    v_curr = self.realized_vol.iloc[i]**2
                    v_prev = self.realized_vol.iloc[i-1]**2
                    
                    # 波動率過程的似然貢獻
                    vol_drift = kappa * (theta - v_prev) * self.dt
                    vol_var = sigma_v**2 * v_prev * self.dt
                    
                    if vol_var > 0:
                        vol_penalty += stats.norm.logpdf(
                            v_curr - v_prev, vol_drift, np.sqrt(vol_var)
                        )
                
                log_likelihood += vol_penalty * 0.1  # 降權重
            
            return log_likelihood if np.isfinite(log_likelihood) else -np.inf
            
        except:
            return -np.inf
    
    def _validate_parameters(self):
        """驗證參數的合理性"""
        warnings_list = []
        
        if self.kappa <= 0:
            warnings_list.append("κ (kappa) should be positive for mean reversion")
        
        if self.theta <= 0:
            warnings_list.append("θ (theta) should be positive")
        
        if self.sigma_v <= 0:
            warnings_list.append("σᵥ (sigma_v) should be positive")
        
        if abs(self.rho) >= 1:
            warnings_list.append("ρ (rho) should be between -1 and 1")
        
        if self.v0 <= 0:
            warnings_list.append("v₀ (v0) should be positive")
        
        # Feller 條件
        if 2 * self.kappa * self.theta <= self.sigma_v**2:
            warnings_list.append("Feller condition violated: 2κθ ≤ σᵥ² may lead to negative volatility")
        
        if warnings_list:
            print("Parameter Validation Warnings:")
            for warning in warnings_list:
                print(f"   ⚠️ {warning}")
    
    def simulate_paths(self, n_paths=1000, n_steps=252, start_price=None, random_seed=42):
        """
        Monte Carlo 模擬 Heston 價格和波動率路徑
        
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
        
        print(f"Running Heston Monte Carlo simulation ({n_paths} paths, {n_steps} steps)...")
        
        if start_price is None:
            start_price = self.prices.iloc[-1]
        
        # 設置隨機種子
        np.random.seed(random_seed)
        
        # 時間間隔
        dt = 1/252
        
        # 初始化陣列
        prices = np.zeros((n_paths, n_steps + 1))
        volatilities = np.zeros((n_paths, n_steps + 1))
        
        prices[:, 0] = start_price
        volatilities[:, 0] = self.v0
        
        # 生成相關隨機數
        for t in range(n_steps):
            # 生成獨立隨機數
            z1 = np.random.normal(0, 1, n_paths)
            z2_indep = np.random.normal(0, 1, n_paths)
            
            # 創建相關隨機數
            z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2_indep
            
            # 當前波動率（確保非負）
            v_curr = np.maximum(volatilities[:, t], 1e-8)
            sqrt_v = np.sqrt(v_curr)
            
            # 更新價格（使用歐拉方案）
            prices[:, t+1] = prices[:, t] * np.exp(
                (self.mu/self.dt - 0.5 * v_curr) * dt + sqrt_v * np.sqrt(dt) * z1
            )
            
            # 更新波動率（使用全截斷方案防止負值）
            vol_drift = self.kappa * (self.theta - v_curr) * dt
            vol_diffusion = self.sigma_v * sqrt_v * np.sqrt(dt) * z2
            
            volatilities[:, t+1] = np.maximum(
                v_curr + vol_drift + vol_diffusion,
                1e-8  # 防止負波動率
            )
        
        # 創建時間索引
        time_index = pd.date_range(
            start=self.prices.index[-1],
            periods=n_steps + 1,
            freq='D'
        )
        
        # 創建結果 DataFrame
        price_df = pd.DataFrame(prices.T, index=time_index)
        vol_df = pd.DataFrame(volatilities.T, index=time_index)
        
        self.simulation_results = {
            'prices': price_df,
            'volatilities': vol_df,
            'price_paths': prices,
            'volatility_paths': volatilities,
            'start_price': start_price,
            'n_paths': n_paths,
            'n_steps': n_steps,
            'dt': dt,
            'final_prices': prices[:, -1],
            'final_volatilities': volatilities[:, -1]
        }
        
        # 統計摘要
        final_prices = prices[:, -1]
        final_vols = volatilities[:, -1]
        returns_sim = (final_prices - start_price) / start_price
        
        print("Heston simulation complete")
        print(f"   Starting price: ${start_price:,.2f}")
        print(f"   Final price statistics:")
        print(f"     Mean: ${np.mean(final_prices):,.2f}")
        print(f"     Median: ${np.median(final_prices):,.2f}")
        print(f"     Std dev: ${np.std(final_prices):,.2f}")
        print(f"   Final volatility statistics:")
        print(f"     Mean: {np.sqrt(np.mean(final_vols))*100:.2f}%")
        print(f"     Std dev: {np.sqrt(np.std(final_vols))*100:.2f}%")
        print(f"   Expected return: {np.mean(returns_sim)*100:.2f}%")
        
        return self.simulation_results
    
    def forecast_price(self, time_horizon=30, confidence_levels=[0.95]):
        """
        價格預測（基於 Heston 模型特徵函數）
        
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
        
        print(f"Generating {time_horizon}-day Heston price forecast...")
        
        current_price = self.prices.iloc[-1]
        t = time_horizon / 365
        
        # 使用 Monte Carlo 方法進行預測
        n_sims = 10000
        dt = t / 100
        
        # 模擬價格路徑
        np.random.seed(42)
        final_prices = []
        
        for _ in range(n_sims):
            S = current_price
            V = self.v0
            
            for step in range(100):
                z1 = np.random.normal()
                z2_indep = np.random.normal()
                z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2_indep
                
                # 價格更新
                S = S * np.exp(
                    (self.mu/self.dt - 0.5 * V) * dt + 
                    np.sqrt(max(V, 1e-8)) * np.sqrt(dt) * z1
                )
                
                # 波動率更新
                V = max(
                    V + self.kappa * (self.theta - V) * dt + 
                    self.sigma_v * np.sqrt(max(V, 1e-8)) * np.sqrt(dt) * z2,
                    1e-8
                )
            
            final_prices.append(S)
        
        final_prices = np.array(final_prices)
        
        # 計算統計量
        expected_price = np.mean(final_prices)
        median_price = np.median(final_prices)
        std_price = np.std(final_prices)
        
        forecast_data = {
            'current_price': current_price,
            'time_horizon_days': time_horizon,
            'expected_price': expected_price,
            'median_price': median_price,
            'price_std': std_price,
            'model_type': 'Heston'
        }
        
        # 計算信心區間
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_bound = np.percentile(final_prices, alpha/2 * 100)
            upper_bound = np.percentile(final_prices, (1 - alpha/2) * 100)
            
            forecast_data[f'lower_{int(conf_level*100)}'] = lower_bound
            forecast_data[f'upper_{int(conf_level*100)}'] = upper_bound
        
        self.forecast_results = forecast_data
        
        print("Heston forecast complete")
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
        計算基於 Heston 模型的 VaR 和 CVaR
        
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
        
        print(f"Calculating Heston risk metrics (portfolio: ${portfolio_value:,})...")
        
        # Monte Carlo 模擬收益率分佈
        n_sims = 50000
        t = time_horizon / 365
        dt = t / 10
        
        returns = []
        current_price = self.prices.iloc[-1]
        
        np.random.seed(42)
        for _ in range(n_sims):
            S = current_price
            V = self.v0
            
            for step in range(10):
                z1 = np.random.normal()
                z2_indep = np.random.normal()
                z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2_indep
                
                S = S * np.exp(
                    (self.mu/self.dt - 0.5 * V) * dt + 
                    np.sqrt(max(V, 1e-8)) * np.sqrt(dt) * z1
                )
                
                V = max(
                    V + self.kappa * (self.theta - V) * dt + 
                    self.sigma_v * np.sqrt(max(V, 1e-8)) * np.sqrt(dt) * z2,
                    1e-8
                )
            
            returns.append((S - current_price) / current_price)
        
        returns = np.array(returns)
        
        risk_metrics = {
            'portfolio_value': portfolio_value,
            'time_horizon': time_horizon,
            'model': 'Heston',
            'n_simulations': n_sims
        }
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            
            # VaR
            var_return = np.percentile(returns, alpha * 100)
            var_dollar = abs(portfolio_value * var_return)
            
            # CVaR
            cvar_returns = returns[returns <= var_return]
            cvar_return = np.mean(cvar_returns) if len(cvar_returns) > 0 else var_return
            cvar_dollar = abs(portfolio_value * cvar_return)
            
            risk_metrics[f'VaR_{int(conf_level*100)}'] = var_dollar
            risk_metrics[f'CVaR_{int(conf_level*100)}'] = cvar_dollar
            
            print(f"   {int(conf_level*100)}% VaR ({time_horizon}日): ${var_dollar:,.0f}")
            print(f"   {int(conf_level*100)}% CVaR ({time_horizon}日): ${cvar_dollar:,.0f}")
        
        return risk_metrics
    
    def analyze_volatility_dynamics(self):
        """
        分析波動率動態特性
        
        Returns:
        --------
        dict : 波動率分析結果
        """
        if not self.fitted:
            raise ValueError("必須先調用 estimate_parameters() 估計參數")
        
        print("Analyzing volatility dynamics...")
        
        # 計算波動率統計量
        vol_stats = {
            'current_vol': np.sqrt(self.v0),
            'long_term_vol': np.sqrt(self.theta),
            'mean_reversion_speed': self.kappa,
            'vol_of_vol': self.sigma_v,
            'correlation': self.rho,
            'half_life': np.log(2) / self.kappa if self.kappa > 0 else np.inf
        }
        
        # Feller 條件檢查
        feller_condition = 2 * self.kappa * self.theta > self.sigma_v**2
        vol_stats['feller_condition'] = feller_condition
        vol_stats['feller_value'] = 2 * self.kappa * self.theta - self.sigma_v**2
        
        # 波動率均值回歸時間
        if self.kappa > 0:
            half_life_days = vol_stats['half_life'] * 365
        else:
            half_life_days = np.inf
        
        # 長期分佈統計
        if feller_condition:
            # 穩態分佈為 Gamma 分佈
            shape = 2 * self.kappa * self.theta / self.sigma_v**2
            rate = 2 * self.kappa / self.sigma_v**2
            
            vol_stats['steady_state_mean'] = shape / rate
            vol_stats['steady_state_var'] = shape / rate**2
        
        print("Volatility dynamics analysis complete")
        print(f"   Current volatility: {vol_stats['current_vol']*100:.2f}%")
        print(f"   Long-term mean volatility: {vol_stats['long_term_vol']*100:.2f}%")
        print(f"   Mean reversion speed: {self.kappa:.4f}")
        print(f"   Half-life: {half_life_days:.1f} days")
        print(f"   Vol-price correlation: {self.rho:.3f}")
        print(f"   Feller condition: {'Satisfied' if feller_condition else 'Violated'}")
        
        return vol_stats
    
    def plot_analysis(self, figsize=(16, 12)):
        """
        創建 Heston 模型綜合分析圖表
        
        Parameters:
        -----------
        figsize : tuple
            圖表大小
        """
        if not self.fitted:
            raise ValueError("必須先調用 estimate_parameters() 估計參數")
        
        plt.rcParams['axes.unicode_minus'] = False
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 1. 價格時間序列
        axes[0, 0].plot(self.prices.index, self.prices.values, 'b-', linewidth=1, alpha=0.8)
        axes[0, 0].set_title('Price Time Series')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 實現波動率時間序列
        axes[0, 1].plot(self.realized_vol.index, self.realized_vol.values*100, 
                       'r-', linewidth=1, alpha=0.8)
        axes[0, 1].axhline(y=np.sqrt(self.theta)*100, color='green', 
                          linestyle='--', label=f'Long-term mean: {np.sqrt(self.theta)*100:.1f}%')
        axes[0, 1].axhline(y=np.sqrt(self.v0)*100, color='orange',
                          linestyle=':', label=f'Current: {np.sqrt(self.v0)*100:.1f}%')
        axes[0, 1].set_title('Realized Volatility')
        axes[0, 1].set_ylabel('Volatility (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 模擬價格路徑（如果有）
        if self.simulation_results is not None:
            paths_to_show = min(50, self.simulation_results['n_paths'])
            sample_indices = np.random.choice(
                self.simulation_results['n_paths'], 
                paths_to_show, 
                replace=False
            )
            
            for idx in sample_indices:
                axes[1, 0].plot(
                    self.simulation_results['prices'].index,
                    self.simulation_results['prices'].iloc[:, idx],
                    alpha=0.3, linewidth=0.5, color='blue'
                )
            
            mean_path = self.simulation_results['prices'].mean(axis=1)
            axes[1, 0].plot(mean_path.index, mean_path.values, 
                          'r-', linewidth=2, label='Mean Path')
            
            axes[1, 0].set_title(f'Simulated Price Paths ({paths_to_show} samples)')
            axes[1, 0].set_ylabel('Price')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No simulation results\nRun simulate_paths() first', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Simulated Price Paths')
        
        # 4. 模擬波動率路徑（如果有）
        if self.simulation_results is not None:
            paths_to_show = min(50, self.simulation_results['n_paths'])
            
            for idx in sample_indices:
                vol_path = np.sqrt(self.simulation_results['volatilities'].iloc[:, idx]) * 100
                axes[1, 1].plot(
                    self.simulation_results['volatilities'].index,
                    vol_path,
                    alpha=0.3, linewidth=0.5, color='red'
                )
            
            mean_vol_path = np.sqrt(self.simulation_results['volatilities'].mean(axis=1)) * 100
            axes[1, 1].plot(mean_vol_path.index, mean_vol_path.values, 
                          'k-', linewidth=2, label='Mean Volatility')
            axes[1, 1].axhline(y=np.sqrt(self.theta)*100, color='green', 
                              linestyle='--', alpha=0.7, label='Long-term Mean')
            
            axes[1, 1].set_title(f'Simulated Volatility Paths ({paths_to_show} samples)')
            axes[1, 1].set_ylabel('Volatility (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No simulation results\nRun simulate_paths() first', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Simulated Volatility Paths')
        
        # 5. 收益率 vs 波動率散點圖
        aligned_returns = self.returns.reindex(self.realized_vol.index).dropna()
        aligned_vol = self.realized_vol.reindex(aligned_returns.index).dropna()
        
        if len(aligned_returns) > 10:
            axes[2, 0].scatter(aligned_vol*100, aligned_returns*100, 
                             alpha=0.5, s=10, color='purple')
            
            # 擬合線
            if len(aligned_returns) > 20:
                z = np.polyfit(aligned_vol*100, aligned_returns*100, 1)
                p = np.poly1d(z)
                vol_range = np.linspace(aligned_vol.min()*100, aligned_vol.max()*100, 100)
                axes[2, 0].plot(vol_range, p(vol_range), 'r--', alpha=0.8, 
                              label=f'Slope: {z[0]:.3f}')
                axes[2, 0].legend()
            
            axes[2, 0].set_xlabel('Realized Volatility (%)')
            axes[2, 0].set_ylabel('Returns (%)')
            axes[2, 0].set_title(f'Returns vs Volatility (ρ={self.rho:.3f})')
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'Insufficient aligned data', 
                           ha='center', va='center', transform=axes[2, 0].transAxes)
        
        # 6. 參數敏感性分析
        if self.forecast_results is not None:
            # 顯示預測分佈
            time_horizon = self.forecast_results['time_horizon_days']
            current_price = self.forecast_results['current_price']
            
            # 模擬最終價格分佈
            np.random.seed(42)
            n_sims = 5000
            final_prices = []
            
            t = time_horizon / 365
            for _ in range(n_sims):
                S = current_price
                V = self.v0
                dt = t / 50
                
                for step in range(50):
                    z1 = np.random.normal()
                    z2_indep = np.random.normal()
                    z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2_indep
                    
                    S = S * np.exp((self.mu/self.dt - 0.5 * V) * dt + 
                                  np.sqrt(max(V, 1e-8)) * np.sqrt(dt) * z1)
                    V = max(V + self.kappa * (self.theta - V) * dt + 
                           self.sigma_v * np.sqrt(max(V, 1e-8)) * np.sqrt(dt) * z2, 1e-8)
                
                final_prices.append(S)
            
            axes[2, 1].hist(final_prices, bins=50, density=True, alpha=0.7, 
                           color='lightblue', label=f'{time_horizon}-day Forecast')
            axes[2, 1].axvline(current_price, color='red', linestyle='--', 
                              alpha=0.8, label='Current Price')
            axes[2, 1].axvline(np.mean(final_prices), color='green', linestyle='-', 
                              alpha=0.8, label='Expected Price')
            
            axes[2, 1].set_title('Price Forecast Distribution')
            axes[2, 1].set_xlabel('Price')
            axes[2, 1].set_ylabel('Density')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'No forecast results\nRun forecast_price() first', 
                           ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Price Forecast Distribution')
        
        plt.tight_layout()
        
        # 保存圖表
        import os
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig('results/plots/heston_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

def demo_heston_advanced():
    """
    Heston 隨機波動率模型演示函數
    """
    print("🎯 Heston 隨機波動率模型演示")
    print("請使用 scripts/demo_heston.py 來運行完整演示")

if __name__ == "__main__":
    demo_heston_advanced()
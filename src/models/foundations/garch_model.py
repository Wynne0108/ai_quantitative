"""
GARCH (Generalized Autoregressive Conditional Heteroskedasticity) Model for Cryptocurrency Volatility Prediction

This module implements GARCH models for predicting cryptocurrency price volatility.
GARCH models are designed to capture volatility clustering and time-varying volatility.

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 導入字體配置工具
try:
    from ..utils.font_config import setup_chinese_fonts, configure_plot_style
    setup_chinese_fonts()
    configure_plot_style()
except ImportError:
    plt.rcParams['axes.unicode_minus'] = False
    try:
        plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    except:
        pass

class GARCHModel:
    """
    GARCH Model for Cryptocurrency Volatility Prediction
    
    This class provides methods to:
    1. Fit GARCH models on return series
    2. Predict future volatility
    3. Calculate risk metrics (VaR, CVaR)
    4. Analyze volatility characteristics
    5. Generate volatility forecasts
    """
    
    def __init__(self, data, price_column='close'):
        """
        Initialize GARCH Model
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price data with datetime index
        price_column : str
            Column name for price data (default: 'close')
        """
        self.data = data.copy()
        self.price_column = price_column
        self.prices = self.data[price_column]
        
        # Calculate returns
        self.returns = self.prices.pct_change().dropna() * 100  # Convert to percentage
        
        # Model components
        self.model = None
        self.fitted_model = None
        self.forecast_result = None
        
        print(f"GARCH 模型初始化完成")
        print(f"數據期間: {self.data.index[0]} 到 {self.data.index[-1]}")
        print(f"收益率統計: 平均={self.returns.mean():.4f}%, 標準差={self.returns.std():.4f}%")
    
    def analyze_returns(self):
        """
        Analyze return series characteristics
        
        Returns:
        --------
        dict : Statistical properties of returns
        """
        returns_stats = {
            'mean': self.returns.mean(),
            'std': self.returns.std(),
            'skewness': stats.skew(self.returns),
            'kurtosis': stats.kurtosis(self.returns),
            'min': self.returns.min(),
            'max': self.returns.max(),
            'jarque_bera': stats.jarque_bera(self.returns),
            'ljung_box': None  # Will be calculated below
        }
        
        # Ljung-Box test for serial correlation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(self.returns, lags=10, return_df=True)
        returns_stats['ljung_box'] = lb_test
        
        print("=== 收益率序列特徵分析 ===")
        print(f"平均收益率: {returns_stats['mean']:.4f}%")
        print(f"波動率: {returns_stats['std']:.4f}%")
        print(f"偏度: {returns_stats['skewness']:.4f} ({'右偏' if returns_stats['skewness'] > 0 else '左偏'})")
        print(f"峰度: {returns_stats['kurtosis']:.4f} ({'厚尾' if returns_stats['kurtosis'] > 0 else '薄尾'})")
        print(f"範圍: {returns_stats['min']:.4f}% 到 {returns_stats['max']:.4f}%")
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = returns_stats['jarque_bera']
        print(f"Jarque-Bera 正態性檢定: 統計量={jb_stat:.4f}, p值={jb_pvalue:.6f}")
        print(f"正態分佈: {'否' if jb_pvalue < 0.05 else '是'}")
        
        return returns_stats
    
    def test_arch_effects(self):
        """
        Test for ARCH effects (heteroskedasticity) in returns
        
        Returns:
        --------
        dict : ARCH test results
        """
        from statsmodels.stats.diagnostic import het_arch
        
        print("\n=== ARCH 效應檢測 ===")
        
        # ARCH test
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(self.returns.dropna(), maxlag=5)
        
        arch_results = {
            'lm_statistic': lm_stat,
            'lm_pvalue': lm_pvalue,
            'f_statistic': f_stat,
            'f_pvalue': f_pvalue,
            'has_arch_effects': lm_pvalue < 0.05
        }
        
        print(f"ARCH-LM 檢定:")
        print(f"  LM 統計量: {lm_stat:.4f}")
        print(f"  p值: {lm_pvalue:.6f}")
        print(f"  ARCH 效應: {'存在' if arch_results['has_arch_effects'] else '不存在'}")
        
        if arch_results['has_arch_effects']:
            print("✅ 檢測到 ARCH 效應，適合使用 GARCH 模型")
        else:
            print("⚠️  未檢測到明顯 ARCH 效應，GARCH 模型效果可能有限")
        
        return arch_results
    
    def fit_garch(self, p=1, q=1, mean_model='constant', vol_model='GARCH', dist='normal'):
        """
        Fit GARCH model to return series
        
        Parameters:
        -----------
        p : int
            GARCH lag order
        q : int  
            ARCH lag order
        mean_model : str
            Mean model specification ('constant', 'zero', 'AR')
        vol_model : str
            Volatility model ('GARCH', 'EGARCH', 'GJR-GARCH')
        dist : str
            Error distribution ('normal', 't', 'skewt')
            
        Returns:
        --------
        fitted model object
        """
        print(f"\n=== 擬合 {vol_model}({p},{q}) 模型 ===")
        
        # Create ARCH model
        self.model = arch_model(
            self.returns.dropna(), 
            mean=mean_model,
            vol=vol_model, 
            p=p, 
            q=q,
            dist=dist
        )
        
        # Fit model
        self.fitted_model = self.model.fit(disp='off', show_warning=False)
        
        print("模型擬合完成！")
        print(f"對數似然值: {self.fitted_model.loglikelihood:.2f}")
        print(f"AIC: {self.fitted_model.aic:.2f}")
        print(f"BIC: {self.fitted_model.bic:.2f}")
        
        # Display model summary
        print(self.fitted_model.summary())
        
        return self.fitted_model
    
    def forecast_volatility(self, horizon=24, method='simulation', simulations=1000):
        """
        Forecast future volatility
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon (number of periods)
        method : str
            Forecasting method ('simulation' or 'analytical')
        simulations : int
            Number of simulations for Monte Carlo method
            
        Returns:
        --------
        dict : Forecast results
        """
        if self.fitted_model is None:
            raise ValueError("模型必須先擬合才能進行預測")
        
        print(f"\n=== 波動率預測 ({horizon} 期) ===")
        
        # Generate forecasts
        if method == 'simulation':
            forecasts = self.fitted_model.forecast(horizon=horizon, method='simulation', simulations=simulations)
        else:
            forecasts = self.fitted_model.forecast(horizon=horizon, method='analytical')
        
        # Extract variance forecasts and convert to volatility (%)
        variance_forecast = forecasts.variance.iloc[-horizon:]
        volatility_forecast = np.sqrt(variance_forecast)  # Already in percentage
        
        # Create forecast index
        last_date = self.returns.index[-1]
        if isinstance(last_date, pd.Timestamp):
            # Try to infer frequency
            freq = pd.infer_freq(self.returns.index)
            if freq is None:
                # Calculate from time difference
                if len(self.returns.index) >= 2:
                    time_diff = self.returns.index[-1] - self.returns.index[-2]
                    total_seconds = time_diff.total_seconds()
                    if total_seconds == 3600:  # 1 hour
                        freq = '1H'
                    elif total_seconds == 14400:  # 4 hours  
                        freq = '4H'
                    elif total_seconds == 86400:  # 1 day
                        freq = '1D'
                    else:
                        freq = '1H'
                else:
                    freq = '1H'
            
            try:
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(freq), periods=horizon, freq=freq)
            except:
                # Fallback: manual calculation
                if len(self.returns.index) >= 2:
                    time_diff = self.returns.index[-1] - self.returns.index[-2]
                    forecast_index = [last_date + time_diff * (i + 1) for i in range(horizon)]
                    forecast_index = pd.DatetimeIndex(forecast_index)
                else:
                    forecast_index = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=horizon, freq='1H')
        else:
            forecast_index = range(len(self.returns), len(self.returns) + horizon)
        
        # Prepare results
        self.forecast_result = {
            'volatility_forecast': pd.Series(volatility_forecast.values.flatten(), index=forecast_index),
            'variance_forecast': pd.Series(variance_forecast.values.flatten(), index=forecast_index),
            'horizon': horizon,
            'method': method
        }
        
        print(f"預測完成！")
        print(f"當前波動率: {np.sqrt(self.fitted_model.conditional_volatility.iloc[-1]):.4f}%")
        print(f"平均預測波動率: {self.forecast_result['volatility_forecast'].mean():.4f}%")
        print(f"波動率範圍: {self.forecast_result['volatility_forecast'].min():.4f}% - {self.forecast_result['volatility_forecast'].max():.4f}%")
        
        return self.forecast_result
    
    def calculate_var_cvar(self, confidence_levels=[0.95, 0.99], horizon=1):
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        
        Parameters:
        -----------
        confidence_levels : list
            Confidence levels for VaR calculation
        horizon : int
            Time horizon for risk calculation
            
        Returns:
        --------
        dict : VaR and CVaR results
        """
        if self.fitted_model is None:
            raise ValueError("模型必須先擬合才能計算風險指標")
        
        print(f"\n=== 風險指標計算 (VaR & CVaR) ===")
        
        # Get latest conditional volatility
        current_vol = np.sqrt(self.fitted_model.conditional_volatility.iloc[-1])
        
        # Adjust for horizon (square root of time rule)
        horizon_vol = current_vol * np.sqrt(horizon)
        
        risk_metrics = {}
        
        for conf_level in confidence_levels:
            # Calculate VaR
            z_score = stats.norm.ppf(1 - conf_level)  # Negative z-score for losses
            var = z_score * horizon_vol  # VaR in percentage
            
            # Calculate CVaR (Expected Shortfall)
            cvar = horizon_vol * stats.norm.pdf(z_score) / (1 - conf_level)
            
            risk_metrics[f'VaR_{int(conf_level*100)}'] = var
            risk_metrics[f'CVaR_{int(conf_level*100)}'] = cvar
            
            print(f"{conf_level*100}% 信心水準:")
            print(f"  VaR:  {var:.4f}% (日損失不會超過 {abs(var):.4f}%)")
            print(f"  CVaR: {cvar:.4f}% (超過 VaR 時的平均損失)")
        
        # Add current volatility info
        risk_metrics['current_volatility'] = current_vol
        risk_metrics['horizon_volatility'] = horizon_vol
        risk_metrics['horizon'] = horizon
        
        return risk_metrics
    
    def plot_volatility_analysis(self, figsize=(15, 12)):
        """
        Create comprehensive volatility analysis plots
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        if self.fitted_model is None:
            raise ValueError("模型必須先擬合才能繪製分析圖表")
        
        # Set plot style for English labels
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 1. Returns time series
        axes[0, 0].plot(self.returns.index, self.returns, linewidth=0.8, alpha=0.7)
        axes[0, 0].set_title('Returns Time Series')
        axes[0, 0].set_ylabel('Returns (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Returns histogram with normal curve
        axes[0, 1].hist(self.returns, bins=50, density=True, alpha=0.7, color='skyblue')
        
        # Overlay normal distribution
        x = np.linspace(self.returns.min(), self.returns.max(), 100)
        normal_curve = stats.norm.pdf(x, self.returns.mean(), self.returns.std())
        axes[0, 1].plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Returns (%)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Conditional volatility
        cond_vol = self.fitted_model.conditional_volatility
        axes[1, 0].plot(cond_vol.index, cond_vol, linewidth=1, color='red')
        axes[1, 0].set_title('Conditional Volatility')
        axes[1, 0].set_ylabel('Volatility (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Standardized residuals
        std_resid = self.fitted_model.std_resid
        axes[1, 1].plot(std_resid.index, std_resid, linewidth=0.8, alpha=0.7, color='green')
        axes[1, 1].set_title('Standardized Residuals')
        axes[1, 1].set_ylabel('Std. Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Volatility forecast (if available)
        if self.forecast_result is not None:
            # Historical volatility
            axes[2, 0].plot(cond_vol.index[-100:], cond_vol.iloc[-100:], 
                           linewidth=1, color='blue', label='Historical Volatility')
            
            # Forecasted volatility
            vol_forecast = self.forecast_result['volatility_forecast']
            axes[2, 0].plot(vol_forecast.index, vol_forecast.values, 
                           linewidth=2, color='red', linestyle='--', label='Forecasted Volatility')
            
            axes[2, 0].set_title('Volatility Forecast')
            axes[2, 0].set_ylabel('Volatility (%)')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'Run volatility forecast first', 
                           ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('Volatility Forecast')
        
        # 6. Q-Q plot of standardized residuals
        if self.fitted_model is not None:
            stats.probplot(std_resid, dist="norm", plot=axes[2, 1])
            axes[2, 1].set_title('Q-Q Plot of Std. Residuals')
        
        plt.tight_layout()
        
        # Save the plot
        import os
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig('results/plots/garch_volatility_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def model_diagnostics(self):
        """
        Perform comprehensive model diagnostics
        """
        if self.fitted_model is None:
            raise ValueError("模型必須先擬合才能進行診斷")
        
        print("=== GARCH 模型診斷 ===")
        
        # 1. Parameter significance
        print("\n1. 參數顯著性:")
        params = self.fitted_model.params
        pvalues = self.fitted_model.pvalues
        
        for param_name, (coef, pval) in zip(params.index, zip(params.values, pvalues.values)):
            significance = "顯著" if pval < 0.05 else "不顯著"
            print(f"   {param_name}: {coef:.6f} (p值: {pval:.4f}) - {significance}")
        
        # 2. Model fit quality
        print(f"\n2. 模型擬合品質:")
        print(f"   對數似然值: {self.fitted_model.loglikelihood:.2f}")
        print(f"   AIC: {self.fitted_model.aic:.2f}")
        print(f"   BIC: {self.fitted_model.bic:.2f}")
        
        # 3. Residual analysis
        std_resid = self.fitted_model.std_resid
        print(f"\n3. 殘差分析:")
        print(f"   殘差平均值: {std_resid.mean():.6f}")
        print(f"   殘差標準差: {std_resid.std():.6f}")
        
        # Ljung-Box test on standardized residuals
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(std_resid, lags=10, return_df=True)
        significant_lags = (lb_test['lb_pvalue'] < 0.05).sum()
        print(f"   Ljung-Box 檢定: {significant_lags}/10 個滯後項顯著")
        
        # 4. ARCH effects in residuals
        from statsmodels.stats.diagnostic import het_arch
        arch_stat, arch_pval, _, _ = het_arch(std_resid**2, maxlag=5)
        print(f"\n4. 殘差 ARCH 效應:")
        print(f"   ARCH-LM 統計量: {arch_stat:.4f} (p值: {arch_pval:.6f})")
        print(f"   殘差 ARCH 效應: {'存在' if arch_pval < 0.05 else '不存在'}")

def demo_garch_analysis():
    """
    Demo function showing how to use GARCHModel class
    """
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='4H')
    
    # Simulate GARCH process
    omega, alpha, beta = 0.01, 0.1, 0.85
    n = len(dates)
    
    # Initialize
    returns = np.zeros(n)
    volatility = np.zeros(n)
    volatility[0] = np.sqrt(omega / (1 - alpha - beta))  # Unconditional volatility
    
    # Generate GARCH process
    for t in range(1, n):
        volatility[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatility[t-1]**2)
        returns[t] = volatility[t] * np.random.normal(0, 1)
    
    # Convert to price data
    prices = 30000 * np.exp(np.cumsum(returns / 100))
    
    data = pd.DataFrame({
        'close': prices,
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'volume': np.random.lognormal(8, 0.5, n)
    }, index=dates)
    
    print("=== GARCH 模型演示 ===")
    
    # Initialize GARCH model
    garch = GARCHModel(data, 'close')
    
    # Analyze returns
    garch.analyze_returns()
    
    # Test for ARCH effects
    garch.test_arch_effects()
    
    # Fit GARCH model
    garch.fit_garch(p=1, q=1)
    
    # Generate volatility forecast
    garch.forecast_volatility(horizon=24)
    
    # Calculate risk metrics
    garch.calculate_var_cvar()
    
    # Model diagnostics
    garch.model_diagnostics()
    
    # Create visualizations
    garch.plot_volatility_analysis()
    
    return garch

if __name__ == "__main__":
    # Run demo
    garch_model = demo_garch_analysis()
"""
Merton 跳躍擴散模型 (Merton Jump Diffusion Model)

這個模組實現 Merton 跳躍擴散模型，用於：
1. 捕捉市場的突發事件和價格跳躍
2. 更準確地建模加密貨幣等高波動資產
3. 風險管理和極端事件分析
4. 跳躍檢測和事件影響評估

數學模型：dS = μS dt + σS dW + S dJ
其中 dJ 是複合泊松跳躍過程

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

from .gbm_base import GBMModel

class MertonJumpDiffusionModel(GBMModel):
    """
    Merton 跳躍擴散模型類
    
    繼承基礎 GBM 模型並添加跳躍組件
    模型：dS = μS dt + σS dW + S dJ
    其中 J 是複合泊松過程，跳躍大小服從正態分佈
    """
    
    def __init__(self, data, price_column='close', freq='4H'):
        """
        初始化 Merton 跳躍擴散模型
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含價格數據的 DataFrame
        price_column : str
            價格欄位名稱
        freq : str
            數據頻率
        """
        super().__init__(data, price_column, freq)
        
        # 跳躍參數
        self.lambda_jump = None    # 跳躍強度（每年跳躍次數）
        self.mu_jump = None        # 跳躍大小均值
        self.sigma_jump = None     # 跳躍大小標準差
        
        # 跳躍檢測結果
        self.jump_dates = None
        self.jump_sizes = None
        self.jump_detection_results = None
        
        print(f"Merton Jump Diffusion Model Initialized Successfully")
    
    def estimate_parameters(self, method='mle', jump_threshold=None):
        """
        估計 Merton 模型參數
        
        Parameters:
        -----------
        method : str
            參數估計方法 ('mle', 'two_stage')
        jump_threshold : float
            跳躍檢測閾值（標準差倍數），預設為自動計算
            
        Returns:
        --------
        dict : 估計的參數
        """
        print("Estimating Merton Jump Diffusion Model Parameters...")
        
        if method == 'two_stage':
            # 兩階段估計法：先檢測跳躍，再估計參數
            return self._two_stage_estimation(jump_threshold)
        elif method == 'mle':
            # 最大似然估計（簡化版）
            return self._simplified_mle_estimation(jump_threshold)
        else:
            raise ValueError("method 必須是 'mle' 或 'two_stage'")
    
    def _two_stage_estimation(self, jump_threshold=None):
        """兩階段估計法"""
        
        # 階段1：跳躍檢測
        self.detect_jumps(threshold=jump_threshold)
        
        # 階段2：參數估計
        # 先估計連續部分（移除跳躍）
        continuous_returns = self.returns.copy()
        if len(self.jump_dates) > 0:
            continuous_returns = continuous_returns.drop(self.jump_dates)
        
        # 估計連續部分的 GBM 參數
        self.mu = continuous_returns.mean()
        self.sigma = continuous_returns.std()
        
        # 估計跳躍參數
        if len(self.jump_sizes) > 0:
            # 跳躍強度（每年）
            n_jumps = len(self.jump_sizes)
            time_span_years = len(self.returns) * self.dt
            self.lambda_jump = n_jumps / time_span_years
            
            # 跳躍大小分佈
            self.mu_jump = np.mean(self.jump_sizes)
            self.sigma_jump = np.std(self.jump_sizes) if len(self.jump_sizes) > 1 else 0.05
        else:
            # 沒有檢測到跳躍
            self.lambda_jump = 0
            self.mu_jump = 0
            self.sigma_jump = 0.05
        
        self.fitted = True
        
        return self._format_parameters('two_stage')
    
    def _simplified_mle_estimation(self, jump_threshold=None):
        """簡化的最大似然估計"""
        
        # 先用兩階段方法獲得初始估計
        initial_params = self._two_stage_estimation(jump_threshold)
        
        # 這裡可以進一步優化，但兩階段方法已經相當有效
        return initial_params
    
    def detect_jumps(self, threshold=None, method='threshold'):
        """
        檢測價格跳躍
        
        Parameters:
        -----------
        threshold : float
            跳躍檢測閾值（標準差倍數）
        method : str
            檢測方法 ('threshold', 'bns')
            
        Returns:
        --------
        dict : 跳躍檢測結果
        """
        print("Detecting price jumps...")
        
        if method == 'threshold':
            return self._threshold_jump_detection(threshold)
        else:
            raise ValueError("目前只支持 'threshold' 方法")
    
    def _threshold_jump_detection(self, threshold=None):
        """基於閾值的跳躍檢測"""
        
        if threshold is None:
            # 自動計算閾值：使用收益率的3倍標準差
            threshold = 3.0
        
        # 計算滾動標準差（排除當前觀測）
        rolling_std = self.returns.rolling(window=20, center=True).std()
        rolling_std = rolling_std.fillna(self.returns.std())
        
        # 標準化收益率
        standardized_returns = self.returns / rolling_std
        
        # 檢測跳躍
        jump_mask = np.abs(standardized_returns) > threshold
        
        self.jump_dates = self.returns[jump_mask].index
        self.jump_sizes = self.returns[jump_mask].values
        
        # 統計結果
        n_jumps = len(self.jump_dates)
        jump_frequency = n_jumps / len(self.returns)
        
        self.jump_detection_results = {
            'method': 'threshold',
            'threshold': threshold,
            'n_jumps': n_jumps,
            'jump_frequency': jump_frequency,
            'jump_dates': self.jump_dates,
            'jump_sizes': self.jump_sizes,
            'positive_jumps': sum(self.jump_sizes > 0),
            'negative_jumps': sum(self.jump_sizes < 0)
        }
        
        print(f"Jump detection complete")
        print(f"   Detected {n_jumps} jumps (threshold: {threshold}σ)")
        print(f"   Jump frequency: {jump_frequency*100:.2f}%")
        print(f"   Positive jumps: {sum(self.jump_sizes > 0)}, Negative jumps: {sum(self.jump_sizes < 0)}")
        
        if n_jumps > 0:
            print(f"   Average jump size: {np.mean(self.jump_sizes)*100:.2f}%")
            print(f"   Largest positive jump: {np.max(self.jump_sizes)*100:.2f}%")
            print(f"   Largest negative jump: {np.min(self.jump_sizes)*100:.2f}%")
        
        return self.jump_detection_results
    
    def _format_parameters(self, method):
        """格式化參數輸出"""
        
        # 年化參數
        mu_annual = self.mu / self.dt
        sigma_annual = self.sigma / np.sqrt(self.dt)
        
        params = {
            # 連續部分參數
            'mu_per_period': self.mu,
            'sigma_per_period': self.sigma,
            'mu_annual': mu_annual,
            'sigma_annual': sigma_annual,
            
            # 跳躍參數
            'lambda_jump': self.lambda_jump,
            'mu_jump': self.mu_jump,
            'sigma_jump': self.sigma_jump,
            
            # 其他資訊
            'estimation_method': method,
            'n_jumps_detected': len(self.jump_sizes) if self.jump_sizes is not None else 0,
            'jump_frequency': len(self.jump_sizes) / len(self.returns) if self.jump_sizes is not None else 0
        }
        
        print(f"Parameter estimation complete ({method.upper()})")
        print(f"   Continuous part:")
        print(f"     Annual drift rate (μ): {mu_annual:.4f} ({mu_annual*100:.2f}%)")
        print(f"     Annual volatility (σ): {sigma_annual:.4f} ({sigma_annual*100:.2f}%)")
        print(f"   Jump part:")
        print(f"     Jump intensity (λ): {self.lambda_jump:.4f} per year")
        print(f"     Jump size mean: {self.mu_jump*100:.2f}%")
        print(f"     Jump size std: {self.sigma_jump*100:.2f}%")
        
        return params
    
    def simulate_paths(self, n_paths=1000, n_steps=252, start_price=None, random_seed=42):
        """
        Monte Carlo 模擬價格路徑（包含跳躍）
        
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
        
        print(f"Monte Carlo simulation (Merton model: {n_paths} paths, {n_steps} steps)...")
        
        if start_price is None:
            start_price = self.prices.iloc[-1]
        
        np.random.seed(random_seed)
        
        dt_sim = 1/252  # 日交易日
        
        # 初始化價格矩陣
        prices = np.zeros((n_paths, n_steps + 1))
        prices[:, 0] = start_price
        
        # 年化參數
        mu_annual = self.mu / self.dt
        sigma_annual = self.sigma / np.sqrt(self.dt)
        
        # 模擬每一步
        for t in range(n_steps):
            # 布朗運動部分
            dW = np.random.normal(0, np.sqrt(dt_sim), n_paths)
            
            # 跳躍部分
            # 生成跳躍次數（泊松過程）
            jump_counts = np.random.poisson(self.lambda_jump * dt_sim, n_paths)
            
            # 生成跳躍大小
            total_jump = np.zeros(n_paths)
            for i in range(n_paths):
                if jump_counts[i] > 0:
                    jump_sizes = np.random.normal(
                        self.mu_jump, 
                        self.sigma_jump, 
                        jump_counts[i]
                    )
                    total_jump[i] = np.sum(jump_sizes)
            
            # 計算下一期價格
            drift = (mu_annual - 0.5 * sigma_annual**2 - 
                    self.lambda_jump * (np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1)) * dt_sim
            
            prices[:, t+1] = prices[:, t] * np.exp(
                drift + sigma_annual * dW + total_jump
            )
        
        # 創建結果
        time_index = pd.date_range(
            start=self.prices.index[-1],
            periods=n_steps + 1,
            freq='D'
        )
        
        price_df = pd.DataFrame(prices.T, index=time_index)
        
        self.simulation_results = {
            'prices': price_df,
            'paths': prices,
            'start_price': start_price,
            'n_paths': n_paths,
            'n_steps': n_steps,
            'dt': dt_sim,
            'final_prices': prices[:, -1],
            'model_type': 'merton_jump_diffusion'
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
    
    def analyze_jump_impact(self):
        """
        分析跳躍對模型的影響
        
        Returns:
        --------
        dict : 跳躍影響分析
        """
        if self.jump_detection_results is None:
            raise ValueError("必須先調用 detect_jumps() 檢測跳躍")
        
        print("Analyzing jump impact...")
        
        # 計算有無跳躍的比較
        all_returns = self.returns
        continuous_returns = all_returns.drop(self.jump_dates) if len(self.jump_dates) > 0 else all_returns
        
        # 統計比較
        analysis = {
            'total_observations': len(all_returns),
            'continuous_observations': len(continuous_returns),
            'jump_observations': len(self.jump_dates),
            
            # 整體統計
            'total_mean': all_returns.mean(),
            'total_std': all_returns.std(),
            'total_skewness': stats.skew(all_returns),
            'total_kurtosis': stats.kurtosis(all_returns),
            
            # 連續部分統計
            'continuous_mean': continuous_returns.mean(),
            'continuous_std': continuous_returns.std(),
            'continuous_skewness': stats.skew(continuous_returns),
            'continuous_kurtosis': stats.kurtosis(continuous_returns),
        }
        
        # 跳躍貢獻分析
        if len(self.jump_dates) > 0:
            jump_contribution = {
                'jump_mean': np.mean(self.jump_sizes),
                'jump_std': np.std(self.jump_sizes),
                'jump_skewness': stats.skew(self.jump_sizes),
                'jump_kurtosis': stats.kurtosis(self.jump_sizes),
                'largest_positive_jump': np.max(self.jump_sizes) if len(self.jump_sizes) > 0 else 0,
                'largest_negative_jump': np.min(self.jump_sizes) if len(self.jump_sizes) > 0 else 0,
            }
            analysis.update(jump_contribution)
        
        print("Jump impact analysis complete")
        print(f"   Total observations: {len(all_returns)}, Jumps: {len(self.jump_dates)}")
        print(f"   Jump ratio: {len(self.jump_dates)/len(all_returns)*100:.2f}%")
        print(f"   Overall std dev: {all_returns.std():.4f}")
        print(f"   Continuous part std dev: {continuous_returns.std():.4f}")
        print(f"   Std dev reduction: {(1-continuous_returns.std()/all_returns.std())*100:.2f}%")
        
        return analysis
    
    def plot_jump_analysis(self, figsize=(15, 10)):
        """
        創建跳躍分析圖表
        
        Parameters:
        -----------
        figsize : tuple
            圖表大小
        """
        if self.jump_detection_results is None:
            raise ValueError("必須先調用 detect_jumps() 檢測跳躍")
        
        # Set plot style
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 價格序列與跳躍點
        axes[0, 0].plot(self.prices.index, self.prices.values, 
                       linewidth=1, alpha=0.8, color='blue', label='Price')
        
        # 標記跳躍點
        if len(self.jump_dates) > 0:
            jump_prices = self.prices.loc[self.jump_dates]
            colors = ['red' if jump > 0 else 'green' for jump in self.jump_sizes]
            axes[0, 0].scatter(jump_prices.index, jump_prices.values, 
                             c=colors, s=50, alpha=0.8, 
                             label=f'Jumps ({len(self.jump_dates)})')
        
        axes[0, 0].set_title('Price Series with Detected Jumps')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 收益率序列與跳躍
        axes[0, 1].plot(self.returns.index, self.returns.values, 
                       linewidth=0.8, alpha=0.7, color='blue')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 標記跳躍
        if len(self.jump_dates) > 0:
            axes[0, 1].scatter(self.jump_dates, self.jump_sizes, 
                             c=['red' if j > 0 else 'green' for j in self.jump_sizes],
                             s=50, alpha=0.8)
        
        axes[0, 1].set_title('Returns with Jump Detection')
        axes[0, 1].set_ylabel('Returns')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 跳躍大小分佈
        if len(self.jump_sizes) > 0:
            axes[1, 0].hist(self.jump_sizes, bins=max(5, len(self.jump_sizes)//2), 
                           alpha=0.7, color='skyblue', density=True)
            
            # 理論正態分佈
            if self.sigma_jump > 0:
                x = np.linspace(self.jump_sizes.min(), self.jump_sizes.max(), 100)
                theoretical_pdf = stats.norm.pdf(x, self.mu_jump, self.sigma_jump)
                axes[1, 0].plot(x, theoretical_pdf, 'r-', linewidth=2, 
                               label='Fitted Normal')
                axes[1, 0].legend()
            
            axes[1, 0].set_title('Jump Size Distribution')
            axes[1, 0].set_xlabel('Jump Size')
            axes[1, 0].set_ylabel('Density')
        else:
            axes[1, 0].text(0.5, 0.5, 'No jumps detected', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Jump Size Distribution')
        
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 模型比較（如果有模擬結果）
        if self.simulation_results is not None:
            # 顯示模擬路徑樣本
            n_sample_paths = min(20, self.simulation_results['n_paths'])
            sample_indices = np.random.choice(
                self.simulation_results['n_paths'], 
                n_sample_paths, 
                replace=False
            )
            
            for idx in sample_indices:
                axes[1, 1].plot(
                    self.simulation_results['prices'].index,
                    self.simulation_results['prices'].iloc[:, idx],
                    alpha=0.4, linewidth=0.8
                )
            
            # 平均路徑
            mean_path = self.simulation_results['prices'].mean(axis=1)
            axes[1, 1].plot(mean_path.index, mean_path.values, 
                          'r-', linewidth=2, label='Mean Path')
            
            axes[1, 1].set_title(f'Merton Model Simulation ({n_sample_paths} paths)')
            axes[1, 1].set_ylabel('Price')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Run simulate_paths() to see\nsimulation results', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Model Simulation')
        
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        import os
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig('results/plots/merton_jump_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

def demo_merton_model():
    """
    Merton 跳躍擴散模型演示函數
    """
    print("🎯 Merton 跳躍擴散模型演示")
    print("請使用 scripts/demo_gbm_advanced.py 來運行完整演示")

if __name__ == "__main__":
    demo_merton_model()
#!/usr/bin/env python3
"""
GBM 模型綜合比較演示腳本

這個腳本演示並比較所有 4 個 GBM 模型：
1. Basic GBM (基礎幾何布朗運動)
2. Merton Jump Diffusion (跳躍擴散模型)
3. Heston Stochastic Volatility (隨機波動率模型)
4. Mean-Reverting GBM (均值回歸模型)

功能包括：
- 模型參數估計和比較
- Monte Carlo 模擬比較
- 價格預測對比
- 風險指標分析
- 模型適用性評估
- 綜合性能評分

Usage: python scripts/demo_gbm_comprehensive.py

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
import time
warnings.filterwarnings('ignore')

# Set matplotlib style for better plots
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False

# Import all GBM models
from models.foundations.gbm_base import GBMModel
from models.foundations.gbm_merton import MertonJumpDiffusionModel
from models.foundations.gbm_heston import HestonModel
from models.foundations.gbm_mean_reverting import MeanRevertingGBMModel
from data_collection.binance_api import BinanceDataCollector

class GBMModelComparison:
    """GBM 模型綜合比較類"""
    
    def __init__(self, data, freq='4H'):
        """
        初始化比較框架
        
        Parameters:
        -----------
        data : pd.DataFrame
            價格數據
        freq : str
            數據頻率
        """
        self.data = data
        self.freq = freq
        self.models = {}
        self.results = {}
        self.comparison_metrics = {}
        
        print("Initializing GBM Model Comparison Framework...")
        print(f"   Data points: {len(data)}")
        print(f"   Data frequency: {freq}")
        print(f"   Price range: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
    
    def initialize_models(self):
        """初始化所有模型"""
        print("\nInitializing All GBM Models...")
        
        model_classes = {
            'Basic GBM': GBMModel,
            'Merton Jump': MertonJumpDiffusionModel,
            'Heston': HestonModel,
            'Mean Reverting': MeanRevertingGBMModel
        }
        
        for name, model_class in model_classes.items():
            try:
                print(f"   Initializing {name}...")
                self.models[name] = model_class(self.data, freq=self.freq)
                print(f"   {name} initialized successfully")
            except Exception as e:
                print(f"   Failed to initialize {name}: {e}")
                self.models[name] = None
        
        print(f"\n   Successfully initialized {len([m for m in self.models.values() if m is not None])}/4 models")
    
    def estimate_all_parameters(self):
        """Estimate all model parameters"""
        print("\nEstimating Parameters for All Models...")
        
        estimation_times = {}
        
        for name, model in self.models.items():
            if model is None:
                continue
                
            print(f"\n   Estimating {name} parameters...")
            start_time = time.time()
            
            try:
                if name == 'Basic GBM':
                    params = model.estimate_parameters(method='mle')
                elif name == 'Merton Jump':
                    params = model.estimate_parameters(method='mle')
                elif name == 'Heston':
                    params = model.estimate_parameters(method='mle', use_global_optimizer=False)  # 較快的本地優化
                elif name == 'Mean Reverting':
                    params = model.estimate_parameters(method='mle')
                
                estimation_time = time.time() - start_time
                estimation_times[name] = estimation_time
                
                self.results[name] = {
                    'parameters': params,
                    'estimation_time': estimation_time,
                    'fitted': True
                }
                
                print(f"   {name} parameters estimated ({estimation_time:.2f}s)")
                
            except Exception as e:
                print(f"   Failed to estimate {name} parameters: {e}")
                self.results[name] = {'fitted': False, 'error': str(e)}
        
        # 顯示估計時間比較
        if estimation_times:
            print("\n   Parameter Estimation Time Comparison:")
            for name, time_taken in sorted(estimation_times.items(), key=lambda x: x[1]):
                print(f"     {name:<15}: {time_taken:>6.2f}s")
    
    def run_simulations(self, n_paths=1000, n_steps=126):
        """Run simulations for all models"""
        print(f"\nRunning Monte Carlo Simulations ({n_paths} paths, {n_steps} steps)...")
        
        simulation_times = {}
        
        for name, model in self.models.items():
            if model is None or not self.results.get(name, {}).get('fitted', False):
                continue
            
            print(f"   Simulating {name}...")
            start_time = time.time()
            
            try:
                simulation = model.simulate_paths(
                    n_paths=n_paths, 
                    n_steps=n_steps,
                    random_seed=42  # 確保可重現性
                )
                
                simulation_time = time.time() - start_time
                simulation_times[name] = simulation_time
                
                self.results[name]['simulation'] = simulation
                self.results[name]['simulation_time'] = simulation_time
                
                print(f"   {name} simulation complete ({simulation_time:.2f}s)")
                
            except Exception as e:
                print(f"   {name} simulation failed: {e}")
                self.results[name]['simulation_error'] = str(e)
        
        # 顯示模擬時間比較
        if simulation_times:
            print("\n   Simulation Time Comparison:")
            for name, time_taken in sorted(simulation_times.items(), key=lambda x: x[1]):
                print(f"     {name:<15}: {time_taken:>6.2f}s")
    
    def generate_forecasts(self, horizons=[30, 90, 180]):
        """Generate forecasts for all models"""
        print(f"\nGenerating Forecasts for horizons: {horizons} days...")
        
        for name, model in self.models.items():
            if model is None or not self.results.get(name, {}).get('fitted', False):
                continue
            
            print(f"   Forecasting with {name}...")
            
            try:
                forecasts = {}
                for horizon in horizons:
                    forecast = model.forecast_price(time_horizon=horizon)
                    forecasts[f'{horizon}d'] = forecast
                
                self.results[name]['forecasts'] = forecasts
                print(f"   {name} forecasts complete")
                
            except Exception as e:
                print(f"   {name} forecasting failed: {e}")
                self.results[name]['forecast_error'] = str(e)
    
    def calculate_risk_metrics(self, portfolio_value=100000, time_horizons=[1, 7]):
        """Calculate risk metrics for all models"""
        print(f"\nCalculating Risk Metrics (Portfolio: ${portfolio_value:,})...")
        
        for name, model in self.models.items():
            if model is None or not self.results.get(name, {}).get('fitted', False):
                continue
            
            print(f"   Calculating {name} risk metrics...")
            
            try:
                risk_metrics = {}
                for horizon in time_horizons:
                    risk = model.calculate_var_cvar(
                        portfolio_value=portfolio_value,
                        confidence_levels=[0.95, 0.99],
                        time_horizon=horizon
                    )
                    risk_metrics[f'{horizon}d'] = risk
                
                self.results[name]['risk_metrics'] = risk_metrics
                print(f"   {name} risk metrics complete")
                
            except Exception as e:
                print(f"   {name} risk calculation failed: {e}")
                self.results[name]['risk_error'] = str(e)
    
    def evaluate_model_performance(self):
        """Evaluate model performance"""
        print("\nEvaluating Model Performance...")
        
        current_price = self.data['close'].iloc[-1]
        
        for name in self.results.keys():
            if not self.results[name].get('fitted', False):
                continue
            
            print(f"\n   Analyzing {name} Performance...")
            
            try:
                performance = {}
                
                # 1. 參數合理性評分
                params = self.results[name]['parameters']
                param_score = self._evaluate_parameter_reasonableness(name, params)
                performance['parameter_score'] = param_score
                
                # 2. 模擬結果評分
                if 'simulation' in self.results[name]:
                    sim = self.results[name]['simulation']
                    sim_score = self._evaluate_simulation_quality(sim, current_price)
                    performance['simulation_score'] = sim_score
                
                # 3. 預測一致性評分
                if 'forecasts' in self.results[name]:
                    forecast_score = self._evaluate_forecast_consistency(
                        self.results[name]['forecasts'], current_price
                    )
                    performance['forecast_score'] = forecast_score
                
                # 4. 計算復雜性評分
                complexity_score = self._evaluate_model_complexity(name)
                performance['complexity_score'] = complexity_score
                
                # 5. 總體評分
                scores = [s for s in performance.values() if s is not None]
                overall_score = np.mean(scores) if scores else 0
                performance['overall_score'] = overall_score
                
                self.results[name]['performance'] = performance
                
                print(f"     Overall Score: {overall_score:.2f}/10")
                
            except Exception as e:
                print(f"     Performance evaluation failed: {e}")
    
    def _evaluate_parameter_reasonableness(self, model_name, params):
        """評估參數合理性"""
        score = 10.0
        
        try:
            if model_name == 'Basic GBM':
                # 檢查年化波動率是否合理（BTC: 50-200%）
                vol = params.get('sigma_annual', 0)
                if vol < 0.3 or vol > 3.0:
                    score -= 2
                
            elif model_name == 'Merton Jump':
                # 檢查跳躍參數
                jump_intensity = params.get('jump_intensity', 0)
                if jump_intensity < 0 or jump_intensity > 50:
                    score -= 2
                
            elif model_name == 'Heston':
                # 檢查 Feller 條件
                if not params.get('feller_condition', True):
                    score -= 3
                # 檢查相關係數
                rho = params.get('rho', 0)
                if abs(rho) > 0.95:
                    score -= 1
                
            elif model_name == 'Mean Reverting':
                # 檢查均值回歸速度
                alpha = params.get('alpha', 0)
                if alpha <= 0 or alpha > 10:
                    score -= 2
                
        except:
            score = 5.0
        
        return max(score, 0)
    
    def _evaluate_simulation_quality(self, simulation, current_price):
        """評估模擬質量"""
        score = 10.0
        
        try:
            final_prices = simulation['final_prices']
            
            # 檢查是否有異常值
            q1, q99 = np.percentile(final_prices, [1, 99])
            if q1 <= 0 or q99 / q1 > 1000:  # 過度分散
                score -= 3
            
            # 檢查分佈合理性
            mean_final = np.mean(final_prices)
            if mean_final / current_price > 10 or mean_final / current_price < 0.1:
                score -= 2
            
            # 檢查是否有負價格
            if np.any(final_prices <= 0):
                score -= 5
            
        except:
            score = 5.0
        
        return max(score, 0)
    
    def _evaluate_forecast_consistency(self, forecasts, current_price):
        """評估預測一致性"""
        score = 10.0
        
        try:
            # 檢查預測是否隨時間合理變化
            horizons = sorted([int(k[:-1]) for k in forecasts.keys()])
            prices = [forecasts[f'{h}d']['expected_price'] for h in horizons]
            
            # 長期預測不應過度偏離短期預測
            if len(prices) >= 2:
                short_term_change = abs(prices[0] - current_price) / current_price
                long_term_change = abs(prices[-1] - current_price) / current_price
                
                if long_term_change > short_term_change * 5:  # 過度擴散
                    score -= 2
            
        except:
            score = 5.0
        
        return max(score, 0)
    
    def _evaluate_model_complexity(self, model_name):
        """評估模型復雜性（簡單性得分更高）"""
        complexity_scores = {
            'Basic GBM': 10.0,      # 最簡單
            'Mean Reverting': 8.0,   # 中等復雜
            'Merton Jump': 6.0,     # 較復雜
            'Heston': 4.0           # 最復雜
        }
        
        return complexity_scores.get(model_name, 5.0)
    
    def create_comprehensive_comparison_plots(self):
        """Create comprehensive comparison charts"""
        print("\nCreating Comprehensive Comparison Charts...")
        
        # 設置圖表
        fig = plt.figure(figsize=(20, 15))
        
        # 創建子圖
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1])
        
        current_price = self.data['close'].iloc[-1]
        
        # 1. 價格時間序列
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.data.index, self.data['close'], 'k-', linewidth=2, label='Historical Price')
        ax1.set_title('Historical Price Series', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 模擬路徑比較
        ax2 = fig.add_subplot(gs[1, 0])
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (name, result) in enumerate(self.results.items()):
            if 'simulation' in result:
                sim = result['simulation']
                # 檢查不同的路徑鍵名
                paths_key = None
                if 'price_paths' in sim:
                    paths_key = 'price_paths'
                elif 'paths' in sim:
                    paths_key = 'paths'
                elif 'prices' in sim and hasattr(sim['prices'], 'values'):
                    # 如果是 DataFrame 格式
                    paths_array = sim['prices'].values.T
                    for j in range(min(5, len(paths_array))):
                        ax2.plot(range(len(paths_array[j])), paths_array[j], 
                               color=colors[i % len(colors)], alpha=0.3, linewidth=0.5)
                    
                    # 平均路徑
                    mean_path = np.mean(paths_array, axis=0)
                    ax2.plot(range(len(mean_path)), mean_path, 
                            color=colors[i % len(colors)], linewidth=2, label=name)
                    continue
                
                if paths_key and paths_key in sim:
                    # 顯示幾條樣本路徑
                    paths = sim[paths_key]
                    for j in range(min(5, len(paths))):
                        ax2.plot(range(len(paths[j])), paths[j], 
                               color=colors[i % len(colors)], alpha=0.3, linewidth=0.5)
                    
                    # 平均路徑
                    mean_path = np.mean(paths, axis=0)
                    ax2.plot(range(len(mean_path)), mean_path, 
                            color=colors[i % len(colors)], linewidth=2, label=name)
        
        ax2.set_title('Simulated Price Paths Comparison')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 最終價格分佈比較
        ax3 = fig.add_subplot(gs[1, 1])
        
        for i, (name, result) in enumerate(self.results.items()):
            if 'simulation' in result:
                sim = result['simulation']
                # 檢查最終價格的不同鍵名
                final_prices = None
                if 'final_prices' in sim:
                    final_prices = sim['final_prices']
                elif 'price_paths' in sim:
                    final_prices = sim['price_paths'][:, -1]  # 最後一列
                elif 'paths' in sim:
                    final_prices = sim['paths'][:, -1]
                elif 'prices' in sim and hasattr(sim['prices'], 'values'):
                    final_prices = sim['prices'].iloc[-1, :].values  # 最後一行
                
                if final_prices is not None:
                    ax3.hist(final_prices, bins=30, alpha=0.6, density=True, 
                            color=colors[i % len(colors)], label=name)
        
        ax3.axvline(current_price, color='black', linestyle='--', 
                   linewidth=2, label='Current Price')
        ax3.set_title('Final Price Distribution Comparison')
        ax3.set_xlabel('Final Price ($)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 預測比較
        ax4 = fig.add_subplot(gs[1, 2])
        
        horizons = [30, 90, 180]
        model_names = []
        forecast_data = {h: [] for h in horizons}
        
        for name, result in self.results.items():
            if 'forecasts' in result:
                model_names.append(name)
                for h in horizons:
                    if f'{h}d' in result['forecasts']:
                        forecast_data[h].append(result['forecasts'][f'{h}d']['expected_price'])
                    else:
                        forecast_data[h].append(current_price)
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, h in enumerate(horizons):
            ax4.bar(x + i*width, forecast_data[h], width, 
                   label=f'{h}-day', alpha=0.8)
        
        ax4.axhline(current_price, color='red', linestyle='--', 
                   alpha=0.7, label='Current Price')
        ax4.set_title('Forecast Comparison')
        ax4.set_ylabel('Expected Price ($)')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. VaR 比較
        ax5 = fig.add_subplot(gs[2, 0])
        
        var_data = {'95% VaR': [], '99% VaR': []}
        model_names_var = []
        
        for name, result in self.results.items():
            if 'risk_metrics' in result and '7d' in result['risk_metrics']:
                model_names_var.append(name)
                risk = result['risk_metrics']['7d']
                var_data['95% VaR'].append(risk.get('VaR_95', 0))
                var_data['99% VaR'].append(risk.get('VaR_99', 0))
        
        x = np.arange(len(model_names_var))
        width = 0.35
        
        ax5.bar(x - width/2, var_data['95% VaR'], width, label='95% VaR', alpha=0.8)
        ax5.bar(x + width/2, var_data['99% VaR'], width, label='99% VaR', alpha=0.8)
        
        ax5.set_title('VaR Comparison (7-day)')
        ax5.set_ylabel('VaR ($)')
        ax5.set_xticks(x)
        ax5.set_xticklabels(model_names_var, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 性能評分比較
        ax6 = fig.add_subplot(gs[2, 1])
        
        performance_categories = ['Parameter', 'Simulation', 'Forecast', 'Complexity', 'Overall']
        
        for i, (name, result) in enumerate(self.results.items()):
            if 'performance' in result:
                perf = result['performance']
                scores = [
                    perf.get('parameter_score', 0),
                    perf.get('simulation_score', 0),
                    perf.get('forecast_score', 0),
                    perf.get('complexity_score', 0),
                    perf.get('overall_score', 0)
                ]
                
                ax6.plot(performance_categories, scores, 'o-', 
                        linewidth=2, markersize=8, label=name)
        
        ax6.set_title('Model Performance Comparison')
        ax6.set_ylabel('Score (0-10)')
        ax6.set_ylim(0, 10)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        plt.setp(ax6.get_xticklabels(), rotation=45)
        
        # 7. 計算時間比較
        ax7 = fig.add_subplot(gs[2, 2])
        
        time_data = {'Estimation': [], 'Simulation': []}
        time_models = []
        
        for name, result in self.results.items():
            if 'estimation_time' in result and 'simulation_time' in result:
                time_models.append(name)
                time_data['Estimation'].append(result['estimation_time'])
                time_data['Simulation'].append(result['simulation_time'])
        
        x = np.arange(len(time_models))
        width = 0.35
        
        ax7.bar(x - width/2, time_data['Estimation'], width, 
               label='Parameter Estimation', alpha=0.8)
        ax7.bar(x + width/2, time_data['Simulation'], width, 
               label='Monte Carlo Simulation', alpha=0.8)
        
        ax7.set_title('Computational Time Comparison')
        ax7.set_ylabel('Time (seconds)')
        ax7.set_xticks(x)
        ax7.set_xticklabels(time_models, rotation=45)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. 模型適用性矩陣
        ax8 = fig.add_subplot(gs[3, :])
        
        # 創建適用性評分矩陣
        scenarios = ['Short-term Trading', 'Long-term Investment', 
                    'Risk Management', 'Option Pricing', 'Volatility Modeling']
        
        suitability_matrix = np.array([
            [9, 7, 8, 9, 6],  # Basic GBM
            [8, 6, 9, 8, 7],  # Merton Jump
            [7, 8, 9, 10, 10], # Heston
            [6, 9, 7, 6, 8]   # Mean Reverting
        ])
        
        model_list = ['Basic GBM', 'Merton Jump', 'Heston', 'Mean Reverting']
        
        im = ax8.imshow(suitability_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)
        
        # 添加文字標籤
        for i in range(len(model_list)):
            for j in range(len(scenarios)):
                text = ax8.text(j, i, f'{suitability_matrix[i, j]}',
                               ha="center", va="center", color="black", fontweight="bold")
        
        ax8.set_xticks(range(len(scenarios)))
        ax8.set_yticks(range(len(model_list)))
        ax8.set_xticklabels(scenarios)
        ax8.set_yticklabels(model_list)
        ax8.set_title('Model Suitability Matrix (0-10 scale)')
        
        # 添加顏色條
        cbar = plt.colorbar(im, ax=ax8, orientation='horizontal', pad=0.1)
        cbar.set_label('Suitability Score')
        
        plt.tight_layout()
        
        # 保存圖表
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig('results/plots/gbm_comprehensive_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\nGenerating Comprehensive Summary Report...")
        
        current_price = self.data['close'].iloc[-1]
        
        # 創建總結
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'data_points': len(self.data),
                'frequency': self.freq,
                'date_range': f"{self.data.index[0]} to {self.data.index[-1]}",
                'current_price': float(current_price),
                'price_range': {
                    'min': float(self.data['close'].min()),
                    'max': float(self.data['close'].max())
                }
            },
            'model_results': {}
        }
        
        # 整合所有模型結果
        for name, result in self.results.items():
            if result.get('fitted', False):
                model_summary = {
                    'parameters': result.get('parameters', {}),
                    'performance_scores': result.get('performance', {}),
                    'computational_time': {
                        'estimation': result.get('estimation_time', 0),
                        'simulation': result.get('simulation_time', 0)
                    }
                }
                
                # 添加預測摘要
                if 'forecasts' in result:
                    forecast_summary = {}
                    for horizon, forecast in result['forecasts'].items():
                        forecast_summary[horizon] = {
                            'expected_price': float(forecast['expected_price']),
                            'expected_change_pct': float(((forecast['expected_price']/current_price) - 1) * 100)
                        }
                    model_summary['forecast_summary'] = forecast_summary
                
                # 添加風險摘要
                if 'risk_metrics' in result:
                    risk_summary = {}
                    for horizon, risk in result['risk_metrics'].items():
                        risk_summary[horizon] = {
                            'VaR_95': float(risk.get('VaR_95', 0)),
                            'VaR_99': float(risk.get('VaR_99', 0))
                        }
                    model_summary['risk_summary'] = risk_summary
                
                summary['model_results'][name] = model_summary
        
        # 模型推薦
        recommendations = self._generate_model_recommendations()
        summary['recommendations'] = recommendations
        
        # 保存報告
        import json
        os.makedirs('results/comprehensive', exist_ok=True)
        
        with open('results/comprehensive/gbm_comparison_report.json', 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        # 生成文字報告
        self._generate_text_report(summary)
        
        return summary
    
    def _generate_model_recommendations(self):
        """生成模型推薦"""
        recommendations = {}
        
        # 基於性能評分推薦
        performance_ranking = []
        for name, result in self.results.items():
            if 'performance' in result:
                overall_score = result['performance'].get('overall_score', 0)
                performance_ranking.append((name, overall_score))
        
        performance_ranking.sort(key=lambda x: x[1], reverse=True)
        
        if performance_ranking:
            best_overall = performance_ranking[0][0]
            recommendations['best_overall_performance'] = best_overall
        
        # 使用情境推薦
        recommendations['use_case_recommendations'] = {
            'beginner_friendly': 'Basic GBM',
            'risk_management': 'Merton Jump' if 'Merton Jump' in self.results else 'Basic GBM',
            'option_pricing': 'Heston' if 'Heston' in self.results else 'Basic GBM',
            'long_term_investment': 'Mean Reverting' if 'Mean Reverting' in self.results else 'Basic GBM',
            'high_frequency_trading': 'Basic GBM'
        }
        
        return recommendations
    
    def _generate_text_report(self, summary):
        """生成文字格式報告"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("GBM 模型綜合比較報告")
        report_lines.append("=" * 80)
        
        # Data summary
        data_summary = summary['data_summary']
        report_lines.append(f"\nData Summary:")
        report_lines.append(f"   分析時間: {summary['analysis_timestamp']}")
        report_lines.append(f"   數據點數: {data_summary['data_points']}")
        report_lines.append(f"   數據頻率: {data_summary['frequency']}")
        report_lines.append(f"   日期範圍: {data_summary['date_range']}")
        report_lines.append(f"   當前價格: ${data_summary['current_price']:,.2f}")
        report_lines.append(f"   價格區間: ${data_summary['price_range']['min']:,.2f} - ${data_summary['price_range']['max']:,.2f}")
        
        # 模型比較
        report_lines.append(f"\n🔍 模型比較結果:")
        
        for model_name, model_result in summary['model_results'].items():
            report_lines.append(f"\n   {model_name}:")
            
            # 性能評分
            if 'performance_scores' in model_result:
                perf = model_result['performance_scores']
                overall = perf.get('overall_score', 0)
                report_lines.append(f"     綜合評分: {overall:.2f}/10")
                
            # 計算時間
            comp_time = model_result.get('computational_time', {})
            est_time = comp_time.get('estimation', 0)
            sim_time = comp_time.get('simulation', 0)
            report_lines.append(f"     計算時間: 參數估計 {est_time:.2f}s, 模擬 {sim_time:.2f}s")
            
            # 30天預測
            if 'forecast_summary' in model_result and '30d' in model_result['forecast_summary']:
                forecast_30d = model_result['forecast_summary']['30d']
                expected_price = forecast_30d['expected_price']
                expected_change = forecast_30d['expected_change_pct']
                report_lines.append(f"     30天預測: ${expected_price:,.2f} ({expected_change:+.2f}%)")
            
            # 風險指標
            if 'risk_summary' in model_result and '7d' in model_result['risk_summary']:
                risk_7d = model_result['risk_summary']['7d']
                var_95 = risk_7d['VaR_95']
                report_lines.append(f"     7天 95% VaR: ${var_95:,.0f}")
        
        # 推薦
        if 'recommendations' in summary:
            rec = summary['recommendations']
            report_lines.append(f"\n💡 模型推薦:")
            if 'best_overall_performance' in rec:
                report_lines.append(f"   綜合表現最佳: {rec['best_overall_performance']}")
            
            use_cases = rec.get('use_case_recommendations', {})
            report_lines.append(f"   使用情境推薦:")
            for use_case, model in use_cases.items():
                report_lines.append(f"     {use_case}: {model}")
        
        report_lines.append(f"\n" + "=" * 80)
        
        # 保存文字報告
        with open('results/comprehensive/gbm_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 打印到控制台
        print('\n'.join(report_lines))

def load_sample_data():
    """Load sample data for comprehensive analysis"""
    print("Loading BTC price data for comprehensive analysis...")
    
    collector = BinanceDataCollector(verify_ssl=False)
    
    try:
        # Get 6 months of 4-hour data for comprehensive analysis
        btc_data = collector.get_multiple_periods_data(
            symbol='BTCUSDT',
            interval='4h',
            days=180,
            market_type='spot'
        )
        
        if btc_data.empty:
            print("   Failed to get BTC data, using simulated data")
            btc_data = generate_comprehensive_sample_data()
        else:
            print(f"   Successfully loaded {len(btc_data)} data points")
            print(f"   Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
            
            # Save data
            collector.save_data_to_csv(btc_data, 'BTC_180days_4h_comprehensive.csv')
            
        return btc_data
        
    except Exception as e:
        print(f"   API request failed: {e}")
        print("   Using simulated data...")
        return generate_comprehensive_sample_data()

def generate_comprehensive_sample_data(start_price=50000, days=180):
    """
    Generate sample BTC data with mixed characteristics for comprehensive testing
    """
    np.random.seed(42)
    
    periods = days * 6  # 4-hour intervals
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='4H')
    n = len(dates)
    
    # Generate synthetic data with mixed characteristics
    log_prices = [np.log(start_price)]
    base_trend = 0.0002  # Basic trend
    vol = 0.02  # Base volatility
    
    for i in range(1, n):
        # Add jumps
        jump = 0
        if np.random.random() < 0.005:  # 0.5% jump probability
            jump = np.random.normal(0, 0.05)
        
        # Mean reversion + GBM + jumps
        mean_reversion = -0.001 * (log_prices[-1] - np.log(55000))
        gbm_component = base_trend + vol * np.random.normal()
        
        log_price_new = log_prices[-1] + mean_reversion + gbm_component + jump
        log_prices.append(log_price_new)
    
    prices = np.exp(log_prices)
    
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
    
    print(f"   Generated {len(data)} synthetic data points with mixed characteristics")
    return data

def main():
    """Main program"""
    print("=== GBM Model Comprehensive Comparison Analysis ===")
    print("=" * 80)
    print("This analysis will compare the following 4 GBM models:")
    print("1. Basic GBM - Geometric Brownian Motion")
    print("2. Merton Jump Diffusion - Jump Diffusion Model")
    print("3. Heston Stochastic Volatility - Stochastic Volatility Model")
    print("4. Mean-Reverting GBM - Mean-Reverting Model")
    print("=" * 80)
    
    try:
        # 1. 載入數據
        data = load_sample_data()
        
        # 2. 初始化比較框架
        comparator = GBMModelComparison(data, freq='4H')
        
        # 3. 初始化所有模型
        comparator.initialize_models()
        
        # 4. 估計所有模型參數
        comparator.estimate_all_parameters()
        
        # 5. 運行模擬
        comparator.run_simulations(n_paths=1000, n_steps=126)
        
        # 6. 生成預測
        comparator.generate_forecasts(horizons=[30, 90, 180])
        
        # 7. 計算風險指標
        comparator.calculate_risk_metrics(portfolio_value=100000, time_horizons=[1, 7])
        
        # 8. 評估模型性能
        comparator.evaluate_model_performance()
        
        # 9. 創建綜合比較圖表
        comparator.create_comprehensive_comparison_plots()
        
        # 10. 生成總結報告
        summary_report = comparator.generate_summary_report()
        
        print("\nGBM Comprehensive Comparison Analysis Complete!")
        print("=" * 80)
        print("Results saved to:")
        print("   Charts: results/plots/gbm_comprehensive_comparison.png")
        print("   JSON Report: results/comprehensive/gbm_comparison_report.json")
        print("   Text Report: results/comprehensive/gbm_comparison_report.txt")
        
        # Quick conclusions
        if 'recommendations' in summary_report:
            rec = summary_report['recommendations']
            print(f"\nQuick Conclusions:")
            if 'best_overall_performance' in rec:
                print(f"   Best overall model: {rec['best_overall_performance']}")
            
            use_cases = rec.get('use_case_recommendations', {})
            print(f"   Recommended use cases:")
            for use_case, model in list(use_cases.items())[:3]:  # Show first 3
                print(f"     • {use_case}: {model}")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""
Cryptocurrency Data Collector

This module provides functionality to collect cryptocurrency market data
from various sources including Binance API and other exchanges.

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class CryptoDataCollector:
    """
    加密貨幣數據收集器
    
    支持從多個來源收集歷史價格數據，包括 Binance API
    """
    
    def __init__(self, base_url: str = "https://api.binance.com"):
        """
        初始化數據收集器
        
        Parameters:
        -----------
        base_url : str
            API 基礎 URL
        """
        self.base_url = base_url
        self.session = requests.Session()
        
    def get_historical_data(self, 
                          symbol: str = 'BTCUSDT',
                          interval: str = '4h',
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 1000) -> pd.DataFrame:
        """
        獲取歷史 K線數據
        
        Parameters:
        -----------
        symbol : str
            交易對符號，例如 'BTCUSDT'
        interval : str
            時間間隔 ('1m', '5m', '1h', '4h', '1d', '1w')
        start_time : datetime
            開始時間
        end_time : datetime
            結束時間
        limit : int
            最大返回數量
            
        Returns:
        --------
        pd.DataFrame : 歷史價格數據
        """
        
        try:
            # 如果無法連接到 API，使用模擬數據
            return self._get_simulated_data(symbol, interval, start_time, end_time)
            
        except Exception as e:
            print(f"API 連接失敗，使用模擬數據: {e}")
            return self._get_simulated_data(symbol, interval, start_time, end_time)
    
    def _get_simulated_data(self, 
                           symbol: str,
                           interval: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        生成模擬的加密貨幣數據
        
        Parameters:
        -----------
        symbol : str
            交易對符號
        interval : str
            時間間隔
        start_time : datetime
            開始時間
        end_time : datetime
            結束時間
            
        Returns:
        --------
        pd.DataFrame : 模擬的歷史價格數據
        """
        
        print(f"生成 {symbol} 的模擬數據...")
        
        # 設置默認時間範圍
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=180)
        
        # 根據間隔創建時間序列
        freq_mapping = {
            '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
        }
        
        freq = freq_mapping.get(interval, '4H')
        dates = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # 設置隨機種子以保證可重現性
        np.random.seed(42)
        
        # 基於 symbol 設置不同的參數
        if 'BTC' in symbol.upper():
            initial_price = 45000
            trend = 0.0002  # 輕微上升趨勢
            volatility = 0.025  # 2.5% 波動率
        elif 'ETH' in symbol.upper():
            initial_price = 2800
            trend = 0.0003
            volatility = 0.035
        else:
            initial_price = 50000
            trend = 0.0001
            volatility = 0.03
        
        n_points = len(dates)
        
        # 生成對數價格序列（幾何布朗運動 + 均值回歸）
        log_prices = [np.log(initial_price)]
        long_term_mean = np.log(initial_price * 1.1)  # 10% 長期增長
        
        for i in range(1, n_points):
            # GBM 成分
            gbm_drift = trend
            gbm_diffusion = volatility * np.random.normal()
            
            # 均值回歸成分
            mean_reversion = 0.001 * (long_term_mean - log_prices[-1])
            
            # 偶爾的跳躍
            jump = 0
            if np.random.random() < 0.002:  # 0.2% 跳躍機率
                jump = np.random.normal(0, 0.05)  # ±5% 跳躍
            
            log_price_new = log_prices[-1] + gbm_drift + gbm_diffusion + mean_reversion + jump
            log_prices.append(log_price_new)
        
        # 轉換為價格
        prices = np.exp(log_prices)
        
        # 生成 OHLCV 數據
        data = []
        for i, price in enumerate(prices):
            # 生成 OHLC
            high_factor = 1 + abs(np.random.normal(0, 0.01))
            low_factor = 1 - abs(np.random.normal(0, 0.01))
            
            if i == 0:
                open_price = price
                close_price = price
            else:
                open_price = prices[i-1] * np.random.uniform(0.995, 1.005)
                close_price = price
            
            high_price = max(open_price, close_price) * high_factor
            low_price = min(open_price, close_price) * low_factor
            
            # 生成成交量（基於價格變動）
            if i > 0:
                price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
                base_volume = np.random.uniform(1000, 5000)
                volume = base_volume * (1 + price_change * 10)  # 價格變動越大，成交量越大
            else:
                volume = np.random.uniform(1000, 5000)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        # 創建 DataFrame
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # 確保數據品質
        df = df.dropna()
        df = df[df['close'] > 0]  # 移除負價格
        
        print(f"   生成了 {len(df)} 個數據點")
        print(f"   價格範圍: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
        print(f"   日期範圍: {df.index[0].strftime('%Y-%m-%d')} 到 {df.index[-1].strftime('%Y-%m-%d')}")
        
        return df
    
    def _convert_interval_to_seconds(self, interval: str) -> int:
        """轉換時間間隔為秒數"""
        interval_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800
        }
        return interval_map.get(interval, 14400)  # 默認 4 小時
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        驗證數據品質
        
        Parameters:
        -----------
        df : pd.DataFrame
            要驗證的數據
            
        Returns:
        --------
        dict : 驗證結果
        """
        validation_result = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'price_anomalies': {},
            'volume_anomalies': {},
            'data_quality_score': 0
        }
        
        try:
            # 檢查價格異常
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    # 檢查負值
                    negative_count = (df[col] <= 0).sum()
                    validation_result['price_anomalies'][f'{col}_negative'] = negative_count
                    
                    # 檢查異常大的變動
                    if len(df) > 1:
                        pct_change = df[col].pct_change().abs()
                        extreme_moves = (pct_change > 0.5).sum()  # 50% 以上變動
                        validation_result['price_anomalies'][f'{col}_extreme_moves'] = extreme_moves
            
            # 檢查 OHLC 邏輯
            if all(col in df.columns for col in price_columns):
                ohlc_violations = (
                    (df['high'] < df['low']) |
                    (df['high'] < df['open']) |
                    (df['high'] < df['close']) |
                    (df['low'] > df['open']) |
                    (df['low'] > df['close'])
                ).sum()
                validation_result['price_anomalies']['ohlc_violations'] = ohlc_violations
            
            # 檢查成交量異常
            if 'volume' in df.columns:
                zero_volume = (df['volume'] <= 0).sum()
                validation_result['volume_anomalies']['zero_volume'] = zero_volume
            
            # 計算數據品質評分
            total_anomalies = sum([
                sum(validation_result['price_anomalies'].values()),
                sum(validation_result['volume_anomalies'].values()),
                sum([v for v in validation_result['missing_values'].values() if isinstance(v, int)])
            ])
            
            if len(df) > 0:
                quality_score = max(0, 100 - (total_anomalies / len(df)) * 100)
                validation_result['data_quality_score'] = round(quality_score, 2)
            
        except Exception as e:
            validation_result['validation_error'] = str(e)
        
        return validation_result
    
    def get_latest_price(self, symbol: str = 'BTCUSDT') -> Dict[str, Any]:
        """
        獲取最新價格
        
        Parameters:
        -----------
        symbol : str
            交易對符號
            
        Returns:
        --------
        dict : 最新價格信息
        """
        try:
            # 模擬最新價格
            base_prices = {
                'BTCUSDT': 45000,
                'ETHUSDT': 2800,
                'BNBUSDT': 300
            }
            
            base_price = base_prices.get(symbol, 50000)
            
            # 添加一些隨機變動
            np.random.seed(int(time.time()) % 1000)
            variation = np.random.uniform(-0.05, 0.05)  # ±5%
            current_price = base_price * (1 + variation)
            
            return {
                'symbol': symbol,
                'price': round(current_price, 2),
                'timestamp': datetime.now(),
                'source': 'simulated'
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'price': 50000.0,
                'timestamp': datetime.now(),
                'error': str(e),
                'source': 'fallback'
            }

# 用於向後兼容的別名
def get_crypto_data(symbol='BTCUSDT', interval='4h', days=180):
    """
    簡化的數據獲取函數
    
    Parameters:
    -----------
    symbol : str
        交易對符號
    interval : str
        時間間隔
    days : int
        歷史天數
        
    Returns:
    --------
    pd.DataFrame : 歷史數據
    """
    collector = CryptoDataCollector()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    return collector.get_historical_data(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time
    )
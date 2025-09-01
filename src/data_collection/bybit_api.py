"""
Bybit API Data Collection Module

This module provides functions to collect cryptocurrency market data from Bybit API.
It supports both spot and derivatives market data collection.

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

class BybitDataCollector:
    """
    Bybit API Data Collector for Cryptocurrency Market Data
    
    This class provides methods to:
    1. Fetch historical kline data from Bybit
    2. Get real-time market data
    3. Handle rate limiting and error handling
    4. Format data for ARIMA model usage
    """
    
    def __init__(self, verify_ssl=False):
        """Initialize Bybit Data Collector"""
        self.base_url = "https://api.bybit.com"
        self.session = requests.Session()
        
        # SSL verification setting
        self.verify_ssl = verify_ssl
        if not verify_ssl:
            # Disable SSL warnings
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Rate limiting parameters
        self.last_request_time = 0
        self.min_request_interval = 0.12  # 120ms between requests (Bybit is more strict)
        
        # Interval mapping for Bybit
        self.interval_mapping = {
            '1': '1m', '3': '3m', '5': '5m', '15': '15m', '30': '30m',
            '60': '1h', '120': '2h', '240': '4h', '360': '6h', '720': '12h',
            'D': '1d', 'W': '1w', 'M': '1M'
        }
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting to avoid API restrictions"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: dict = None) -> dict:
        """
        Make API request with error handling and rate limiting
        
        Parameters:
        -----------
        url : str
            API endpoint URL
        params : dict
            Request parameters
            
        Returns:
        --------
        dict : API response
        """
        self._wait_for_rate_limit()
        
        try:
            response = self.session.get(url, params=params, timeout=30, verify=self.verify_ssl)
            response.raise_for_status()
            data = response.json()
            
            # Check Bybit API response format
            if data.get('retCode') == 0:
                return data.get('result', {})
            else:
                print(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
                return {}
                
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return {}
    
    def _convert_interval_to_bybit(self, interval: str) -> str:
        """
        Convert standard interval to Bybit format
        
        Parameters:
        -----------
        interval : str
            Standard interval (e.g., '1h', '1d')
            
        Returns:
        --------
        str : Bybit interval format
        """
        bybit_intervals = {
            '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '2h': '120', '4h': '240', '6h': '360', '12h': '720',
            '1d': 'D', '1w': 'W', '1M': 'M'
        }
        
        return bybit_intervals.get(interval, '60')  # Default to 1 hour
    
    def get_historical_klines(self,
                            symbol: str,
                            interval: str = '1h',
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: int = 200,
                            category: str = 'spot') -> pd.DataFrame:
        """
        Get historical kline data from Bybit API
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        interval : str
            Kline interval ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M')
        start_time : datetime, optional
            Start time for data collection
        end_time : datetime, optional
            End time for data collection
        limit : int
            Number of klines to retrieve (max 200 for Bybit)
        category : str
            Product category ('spot', 'linear', 'inverse')
            
        Returns:
        --------
        pd.DataFrame : Historical kline data with OHLCV columns
        """
        # Bybit V5 API endpoint
        url = f"{self.base_url}/v5/market/kline"
        
        params = {
            'category': category,
            'symbol': symbol.upper(),
            'interval': self._convert_interval_to_bybit(interval),
            'limit': min(limit, 200)  # Bybit limit is 200
        }
        
        if start_time:
            params['start'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['end'] = int(end_time.timestamp() * 1000)
        
        data = self._make_request(url, params)
        
        if not data or 'list' not in data:
            print(f"Failed to fetch data for {symbol} from Bybit")
            return pd.DataFrame()
        
        klines = data['list']
        
        if not klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        # Bybit returns: [timestamp, open, high, low, close, volume, turnover]
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        df = pd.DataFrame(klines, columns=columns)
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Sort by timestamp (Bybit returns in reverse order)
        df = df.sort_index()
        
        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        return df
    
    def get_multiple_periods_data(self,
                                symbol: str,
                                interval: str = '1h',
                                days: int = 30,
                                category: str = 'spot') -> pd.DataFrame:
        """
        Get historical data for multiple periods by making multiple API calls
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        interval : str
            Kline interval
        days : int
            Number of days of historical data to fetch
        category : str
            Product category
            
        Returns:
        --------
        pd.DataFrame : Combined historical data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        all_data = []
        current_end = end_time
        
        # Calculate how many requests we need (Bybit limit is 200 per request)
        intervals_per_day = {
            '1m': 1440, '3m': 480, '5m': 288, '15m': 96, '30m': 48,
            '1h': 24, '2h': 12, '4h': 6, '6h': 4, '12h': 2, '1d': 1
        }
        
        points_per_day = intervals_per_day.get(interval, 24)
        total_points = days * points_per_day
        requests_needed = (total_points + 199) // 200  # Round up
        
        print(f"Fetching {total_points} data points in {requests_needed} requests")
        
        for i in range(requests_needed):
            batch_start = current_end - timedelta(days=min(days, 200/points_per_day))
            
            print(f"Request {i+1}/{requests_needed}: {batch_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
            
            batch_data = self.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_time=batch_start,
                end_time=current_end,
                limit=200,
                category=category
            )
            
            if not batch_data.empty:
                all_data.append(batch_data)
            
            current_end = batch_start
            
            if current_end <= start_time:
                break
                
            time.sleep(0.2)  # Additional delay between batches
        
        if all_data:
            combined_data = pd.concat(all_data, axis=0)
            # Remove duplicates and sort by timestamp
            combined_data = combined_data.drop_duplicates().sort_index()
            # Filter to exact date range
            combined_data = combined_data[combined_data.index >= start_time]
            print(f"Total data points collected: {len(combined_data)}")
            return combined_data
        else:
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str, category: str = 'spot') -> float:
        """
        Get current price for a symbol
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        category : str
            Product category
            
        Returns:
        --------
        float : Current price
        """
        url = f"{self.base_url}/v5/market/tickers"
        params = {
            'category': category,
            'symbol': symbol.upper()
        }
        
        data = self._make_request(url, params)
        
        if data and 'list' in data and len(data['list']) > 0:
            return float(data['list'][0].get('lastPrice', 0))
        
        return 0.0
    
    def get_24hr_stats(self, symbol: str, category: str = 'spot') -> Dict:
        """
        Get 24-hour ticker statistics
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        category : str
            Product category
            
        Returns:
        --------
        dict : 24-hour statistics
        """
        url = f"{self.base_url}/v5/market/tickers"
        params = {
            'category': category,
            'symbol': symbol.upper()
        }
        
        data = self._make_request(url, params)
        
        if data and 'list' in data and len(data['list']) > 0:
            ticker = data['list'][0]
            return {
                'symbol': ticker.get('symbol'),
                'price_change': float(ticker.get('price24hPcnt', 0)) / 100 * float(ticker.get('lastPrice', 0)),
                'price_change_percent': float(ticker.get('price24hPcnt', 0)),
                'high_price': float(ticker.get('highPrice24h', 0)),
                'low_price': float(ticker.get('lowPrice24h', 0)),
                'volume': float(ticker.get('volume24h', 0)),
                'turnover': float(ticker.get('turnover24h', 0))
            }
        
        return {}
    
    def save_data_to_csv(self, data: pd.DataFrame, filename: str, folder: str = 'data/raw/bybit'):
        """
        Save data to CSV file
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to save
        filename : str
            Output filename
        folder : str
            Output folder path
        """
        import os
        
        # Create folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        filepath = os.path.join(folder, filename)
        data.to_csv(filepath)
        print(f"Data saved to: {filepath}")


def demo_bybit_data_collection():
    """
    Demo function showing how to use BybitDataCollector
    """
    collector = BybitDataCollector()
    
    print("=== Bybit Data Collection Demo ===")
    
    # Example 1: Get current price
    print("\n1. Current BTC price from Bybit:")
    try:
        btc_price = collector.get_current_price('BTCUSDT')
        print(f"BTC/USDT: ${btc_price:,.2f}")
    except Exception as e:
        print(f"Error getting current price: {e}")
    
    # Example 2: Get 24hr statistics
    print("\n2. 24-hour statistics:")
    try:
        stats = collector.get_24hr_stats('BTCUSDT')
        if stats:
            print(f"Price Change: ${stats['price_change']:+,.2f} ({stats['price_change_percent']:+.2f}%)")
            print(f"High: ${stats['high_price']:,.2f}")
            print(f"Low: ${stats['low_price']:,.2f}")
            print(f"Volume: {stats['volume']:,.2f} BTC")
    except Exception as e:
        print(f"Error getting 24hr stats: {e}")
    
    # Example 3: Get historical data
    print("\n3. Historical data (last 3 days, 1-hour intervals):")
    try:
        btc_data = collector.get_multiple_periods_data(
            symbol='BTCUSDT',
            interval='1h',
            days=3,
            category='spot'
        )
        
        if not btc_data.empty:
            print(f"Data period: {btc_data.index[0]} to {btc_data.index[-1]}")
            print(f"Total data points: {len(btc_data)}")
            print("\nData preview:")
            print(btc_data.tail())
            
            # Save data
            collector.save_data_to_csv(btc_data, 'BTCUSDT_1h_3days_bybit.csv')
        else:
            print("No data received")
            
    except Exception as e:
        print(f"Error getting historical data: {e}")
    
    return collector


if __name__ == "__main__":
    # Run demo
    collector = demo_bybit_data_collection()
"""
Binance API Data Collection Module

This module provides functions to collect cryptocurrency market data from Binance API.
It supports both spot and futures market data collection for ARIMA model training.

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class BinanceDataCollector:
    """
    Binance API Data Collector for Cryptocurrency Market Data
    
    This class provides methods to:
    1. Fetch historical kline data
    2. Get real-time market data
    3. Handle rate limiting and error handling
    4. Format data for ARIMA model usage
    """
    
    def __init__(self, verify_ssl=False):
        """Initialize Binance Data Collector"""
        self.base_url = "https://api.binance.com"
        self.futures_base_url = "https://fapi.binance.com"
        self.session = requests.Session()
        
        # SSL verification setting
        self.verify_ssl = verify_ssl
        if not verify_ssl:
            # Disable SSL warnings
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Rate limiting parameters
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
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
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return None
    
    def get_symbol_info(self, symbol: str, market_type: str = 'spot') -> Dict:
        """
        Get symbol information and trading rules
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., 'BTCUSDT')
        market_type : str
            Market type ('spot' or 'futures')
            
        Returns:
        --------
        dict : Symbol information
        """
        if market_type == 'spot':
            url = f"{self.base_url}/api/v3/exchangeInfo"
        else:
            url = f"{self.futures_base_url}/fapi/v1/exchangeInfo"
        
        data = self._make_request(url)
        
        if data:
            for symbol_info in data.get('symbols', []):
                if symbol_info['symbol'] == symbol.upper():
                    return symbol_info
        
        return {}
    
    def get_historical_klines(self, 
                            symbol: str,
                            interval: str = '1h',
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: int = 1000,
                            market_type: str = 'spot') -> pd.DataFrame:
        """
        Get historical kline data from Binance API
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        interval : str
            Kline interval ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
        start_time : datetime, optional
            Start time for data collection
        end_time : datetime, optional
            End time for data collection  
        limit : int
            Number of klines to retrieve (max 1000)
        market_type : str
            Market type ('spot' or 'futures')
            
        Returns:
        --------
        pd.DataFrame : Historical kline data with OHLCV columns
        """
        if market_type == 'spot':
            url = f"{self.base_url}/api/v3/klines"
        else:
            url = f"{self.futures_base_url}/fapi/v1/klines"
        
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        data = self._make_request(url, params)
        
        if not data:
            print(f"Failed to fetch data for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                  'taker_buy_quote_volume', 'ignore']
        
        df = pd.DataFrame(data, columns=columns)
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        return df
    
    def get_multiple_periods_data(self,
                                symbol: str,
                                interval: str = '1h',
                                days: int = 30,
                                market_type: str = 'spot') -> pd.DataFrame:
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
        market_type : str
            Market type ('spot' or 'futures')
            
        Returns:
        --------
        pd.DataFrame : Combined historical data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + timedelta(days=30), end_time)
            
            print(f"Fetching data from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
            
            batch_data = self.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end,
                limit=1000,
                market_type=market_type
            )
            
            if not batch_data.empty:
                all_data.append(batch_data)
            
            current_start = current_end
            time.sleep(0.2)  # Additional delay between batches
        
        if all_data:
            combined_data = pd.concat(all_data, axis=0)
            # Remove duplicates and sort by timestamp
            combined_data = combined_data.drop_duplicates().sort_index()
            print(f"Total data points collected: {len(combined_data)}")
            return combined_data
        else:
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str, market_type: str = 'spot') -> float:
        """
        Get current price for a symbol
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        market_type : str
            Market type ('spot' or 'futures')
            
        Returns:
        --------
        float : Current price
        """
        if market_type == 'spot':
            url = f"{self.base_url}/api/v3/ticker/price"
        else:
            url = f"{self.futures_base_url}/fapi/v1/ticker/price"
        
        params = {'symbol': symbol.upper()}
        data = self._make_request(url, params)
        
        if data and 'price' in data:
            return float(data['price'])
        
        return 0.0
    
    def get_24hr_stats(self, symbol: str, market_type: str = 'spot') -> Dict:
        """
        Get 24-hour ticker statistics
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        market_type : str
            Market type ('spot' or 'futures')
            
        Returns:
        --------
        dict : 24-hour statistics
        """
        if market_type == 'spot':
            url = f"{self.base_url}/api/v3/ticker/24hr"
        else:
            url = f"{self.futures_base_url}/fapi/v1/ticker/24hr"
        
        params = {'symbol': symbol.upper()}
        data = self._make_request(url, params)
        
        if data:
            return {
                'symbol': data.get('symbol'),
                'price_change': float(data.get('priceChange', 0)),
                'price_change_percent': float(data.get('priceChangePercent', 0)),
                'high_price': float(data.get('highPrice', 0)),
                'low_price': float(data.get('lowPrice', 0)),
                'volume': float(data.get('volume', 0)),
                'count': int(data.get('count', 0))
            }
        
        return {}
    
    def save_data_to_csv(self, data: pd.DataFrame, filename: str, folder: str = 'data/raw/binance'):
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
    
    def load_data_from_csv(self, filename: str, folder: str = 'data/raw/binance') -> pd.DataFrame:
        """
        Load data from CSV file
        
        Parameters:
        -----------
        filename : str
            CSV filename
        folder : str
            Data folder path
            
        Returns:
        --------
        pd.DataFrame : Loaded data
        """
        import os
        
        filepath = os.path.join(folder, filename)
        
        if os.path.exists(filepath):
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"Data loaded from: {filepath}")
            return data
        else:
            print(f"File not found: {filepath}")
            return pd.DataFrame()


def demo_binance_data_collection():
    """
    Demo function showing how to use BinanceDataCollector
    """
    collector = BinanceDataCollector()
    
    print("=== Binance Data Collection Demo ===")
    
    # Example 1: Get current price
    print("\n1. Current BTC price:")
    btc_price = collector.get_current_price('BTCUSDT')
    print(f"BTC/USDT: ${btc_price:,.2f}")
    
    # Example 2: Get 24hr statistics
    print("\n2. 24-hour statistics:")
    stats = collector.get_24hr_stats('BTCUSDT')
    if stats:
        print(f"Price Change: ${stats['price_change']:+,.2f} ({stats['price_change_percent']:+.2f}%)")
        print(f"High: ${stats['high_price']:,.2f}")
        print(f"Low: ${stats['low_price']:,.2f}")
        print(f"Volume: {stats['volume']:,.2f} BTC")
    
    # Example 3: Get historical data
    print("\n3. Historical data (last 7 days, 1-hour intervals):")
    btc_data = collector.get_multiple_periods_data(
        symbol='BTCUSDT',
        interval='1h',
        days=7,
        market_type='spot'
    )
    
    if not btc_data.empty:
        print(f"Data period: {btc_data.index[0]} to {btc_data.index[-1]}")
        print(f"Total data points: {len(btc_data)}")
        print("\nData preview:")
        print(btc_data.tail())
        
        # Save data
        collector.save_data_to_csv(btc_data, 'BTCUSDT_1h_7days.csv')
    
    # Example 4: Multiple symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    print(f"\n4. Current prices for multiple symbols:")
    
    for symbol in symbols:
        price = collector.get_current_price(symbol)
        print(f"{symbol}: ${price:,.2f}")
    
    return collector, btc_data


if __name__ == "__main__":
    # Run demo
    collector, data = demo_binance_data_collection()
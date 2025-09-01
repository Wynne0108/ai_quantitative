"""
Unified Data Downloader for Cryptocurrency Market Data

This module provides a unified interface for downloading cryptocurrency data
from multiple exchanges (Binance, Bybit) for quantitative analysis.

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from .binance_api import BinanceDataCollector
from .bybit_api import BybitDataCollector

class UnifiedDataDownloader:
    """
    Unified data downloader that supports multiple cryptocurrency exchanges
    """
    
    def __init__(self, data_folder: str = 'data/raw'):
        """
        Initialize unified data downloader
        
        Parameters:
        -----------
        data_folder : str
            Base folder for storing downloaded data
        """
        self.data_folder = data_folder
        self.binance_collector = BinanceDataCollector()
        self.bybit_collector = BybitDataCollector()
        
        # Create data folders
        self._create_data_folders()
        
        # Supported exchanges
        self.supported_exchanges = ['binance', 'bybit']
        
        # Common trading pairs
        self.popular_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
            'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LTCUSDT'
        ]
        
        # Time intervals mapping
        self.interval_mapping = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }
    
    def _create_data_folders(self):
        """Create necessary data folders"""
        folders = [
            os.path.join(self.data_folder, 'binance'),
            os.path.join(self.data_folder, 'bybit'),
            'data/processed',
            'data/cache'
        ]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
    
    def download_single_pair(self,
                           symbol: str,
                           exchange: str = 'binance',
                           interval: str = '1h',
                           days: int = 30,
                           market_type: str = 'spot',
                           save_to_file: bool = True) -> pd.DataFrame:
        """
        Download data for a single trading pair
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        exchange : str
            Exchange name ('binance' or 'bybit')
        interval : str
            Time interval
        days : int
            Number of days of historical data
        market_type : str
            Market type ('spot' or 'futures')
        save_to_file : bool
            Whether to save data to CSV file
            
        Returns:
        --------
        pd.DataFrame : Downloaded market data
        """
        print(f"📥 Downloading {symbol} from {exchange.upper()}")
        print(f"   Interval: {interval}, Days: {days}, Market: {market_type}")
        
        data = pd.DataFrame()
        
        if exchange.lower() == 'binance':
            try:
                data = self.binance_collector.get_multiple_periods_data(
                    symbol=symbol,
                    interval=interval,
                    days=days,
                    market_type=market_type
                )
                
                if not data.empty:
                    print(f"✅ Successfully downloaded {len(data)} data points")
                    
                    if save_to_file:
                        filename = f"{symbol}_{interval}_{days}days_{market_type}.csv"
                        filepath = os.path.join(self.data_folder, 'binance', filename)
                        data.to_csv(filepath)
                        print(f"💾 Data saved to: {filepath}")
                else:
                    print("❌ No data received")
                    
            except Exception as e:
                print(f"❌ Error downloading from Binance: {e}")
        
        elif exchange.lower() == 'bybit':
            try:
                # Convert market_type for Bybit
                category = 'spot' if market_type == 'spot' else 'linear'
                
                data = self.bybit_collector.get_multiple_periods_data(
                    symbol=symbol,
                    interval=interval,
                    days=days,
                    category=category
                )
                
                if not data.empty:
                    print(f"✅ Successfully downloaded {len(data)} data points")
                    
                    if save_to_file:
                        filename = f"{symbol}_{interval}_{days}days_{market_type}.csv"
                        filepath = os.path.join(self.data_folder, 'bybit', filename)
                        data.to_csv(filepath)
                        print(f"💾 Data saved to: {filepath}")
                else:
                    print("❌ No data received")
                    
            except Exception as e:
                print(f"❌ Error downloading from Bybit: {e}")
                print("🔄 Trying Binance as fallback...")
                return self.download_single_pair(symbol, 'binance', interval, days, market_type, save_to_file)
        
        else:
            print(f"❌ Exchange '{exchange}' not supported")
            
        return data
    
    def download_multiple_pairs(self,
                              symbols: List[str],
                              exchange: str = 'binance',
                              interval: str = '1h',
                              days: int = 30,
                              market_type: str = 'spot',
                              save_to_file: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple trading pairs
        
        Parameters:
        -----------
        symbols : List[str]
            List of trading pair symbols
        exchange : str
            Exchange name
        interval : str
            Time interval
        days : int
            Number of days of historical data
        market_type : str
            Market type
        save_to_file : bool
            Whether to save data to files
            
        Returns:
        --------
        Dict[str, pd.DataFrame] : Dictionary of symbol -> data
        """
        print(f"📥 Downloading {len(symbols)} symbols from {exchange.upper()}")
        
        data_dict = {}
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Processing {symbol}")
            
            try:
                data = self.download_single_pair(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    days=days,
                    market_type=market_type,
                    save_to_file=save_to_file
                )
                
                if not data.empty:
                    data_dict[symbol] = data
                else:
                    print(f"⚠️  Skipping {symbol} due to no data")
                    
                # Rate limiting between requests
                import time
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                continue
        
        print(f"\n✅ Successfully downloaded {len(data_dict)}/{len(symbols)} symbols")
        return data_dict
    
    def download_popular_pairs(self,
                             exchange: str = 'binance',
                             interval: str = '1h',
                             days: int = 30,
                             market_type: str = 'spot') -> Dict[str, pd.DataFrame]:
        """
        Download data for popular trading pairs
        
        Parameters:
        -----------
        exchange : str
            Exchange name
        interval : str
            Time interval
        days : int
            Number of days
        market_type : str
            Market type
            
        Returns:
        --------
        Dict[str, pd.DataFrame] : Dictionary of popular pairs data
        """
        print(f"📥 Downloading popular crypto pairs")
        print(f"Pairs: {', '.join(self.popular_pairs)}")
        
        return self.download_multiple_pairs(
            symbols=self.popular_pairs,
            exchange=exchange,
            interval=interval,
            days=days,
            market_type=market_type
        )
    
    def load_data_from_file(self,
                          symbol: str,
                          exchange: str = 'binance',
                          interval: str = '1h',
                          days: int = 30,
                          market_type: str = 'spot') -> pd.DataFrame:
        """
        Load previously downloaded data from file
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        exchange : str
            Exchange name
        interval : str
            Time interval
        days : int
            Number of days
        market_type : str
            Market type
            
        Returns:
        --------
        pd.DataFrame : Loaded data
        """
        filename = f"{symbol}_{interval}_{days}days_{market_type}.csv"
        filepath = os.path.join(self.data_folder, exchange, filename)
        
        if os.path.exists(filepath):
            print(f"📁 Loading data from: {filepath}")
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"✅ Loaded {len(data)} data points")
            return data
        else:
            print(f"❌ File not found: {filepath}")
            return pd.DataFrame()
    
    def get_data_summary(self, data: pd.DataFrame, symbol: str = "") -> Dict:
        """
        Generate summary statistics for the data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data
        symbol : str
            Symbol name for display
            
        Returns:
        --------
        Dict : Summary statistics
        """
        if data.empty:
            return {}
        
        price_col = 'close'
        returns = data[price_col].pct_change().dropna()
        
        summary = {
            'symbol': symbol,
            'period': f"{data.index[0]} to {data.index[-1]}",
            'data_points': len(data),
            'price_range': {
                'min': float(data[price_col].min()),
                'max': float(data[price_col].max()),
                'current': float(data[price_col].iloc[-1])
            },
            'returns': {
                'mean': float(returns.mean()),
                'std': float(returns.std()),
                'min': float(returns.min()),
                'max': float(returns.max())
            },
            'volume': {
                'mean': float(data['volume'].mean()),
                'total': float(data['volume'].sum())
            }
        }
        
        return summary
    
    def create_dataset_for_arima(self,
                               symbol: str,
                               exchange: str = 'binance',
                               interval: str = '1h',
                               days: int = 30,
                               market_type: str = 'spot',
                               price_column: str = 'close') -> pd.DataFrame:
        """
        Create a clean dataset specifically for ARIMA analysis
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        exchange : str
            Exchange name
        interval : str
            Time interval
        days : int
            Number of days
        market_type : str
            Market type
        price_column : str
            Price column to use
            
        Returns:
        --------
        pd.DataFrame : Clean data for ARIMA
        """
        print(f"🔧 Preparing ARIMA dataset for {symbol}")
        
        # Try to load from file first
        data = self.load_data_from_file(symbol, exchange, interval, days, market_type)
        
        # If no file exists, download fresh data
        if data.empty:
            print("📥 No cached data found, downloading fresh data...")
            data = self.download_single_pair(symbol, exchange, interval, days, market_type)
        
        if data.empty:
            print("❌ No data available")
            return pd.DataFrame()
        
        # Clean the data for ARIMA
        clean_data = data.copy()
        
        # Remove any rows with missing values
        clean_data = clean_data.dropna()
        
        # Ensure price column exists
        if price_column not in clean_data.columns:
            print(f"❌ Price column '{price_column}' not found")
            return pd.DataFrame()
        
        # Remove any zero or negative prices
        clean_data = clean_data[clean_data[price_column] > 0]
        
        # Sort by timestamp
        clean_data = clean_data.sort_index()
        
        print(f"✅ Clean dataset ready: {len(clean_data)} data points")
        print(f"   Period: {clean_data.index[0]} to {clean_data.index[-1]}")
        print(f"   Price range: ${clean_data[price_column].min():.2f} - ${clean_data[price_column].max():.2f}")
        
        return clean_data
    
    def get_current_market_overview(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Get current market overview for multiple symbols
        
        Parameters:
        -----------
        symbols : List[str], optional
            List of symbols (default: popular pairs)
            
        Returns:
        --------
        pd.DataFrame : Market overview
        """
        if symbols is None:
            symbols = self.popular_pairs[:5]  # Top 5 popular pairs
        
        print(f"📊 Getting market overview for: {', '.join(symbols)}")
        
        overview_data = []
        
        for symbol in symbols:
            try:
                # Get current price
                current_price = self.binance_collector.get_current_price(symbol)
                
                # Get 24hr stats
                stats = self.binance_collector.get_24hr_stats(symbol)
                
                if stats:
                    overview_data.append({
                        'Symbol': symbol,
                        'Current_Price': current_price,
                        'Change_24h': stats['price_change'],
                        'Change_24h_Pct': stats['price_change_percent'],
                        'High_24h': stats['high_price'],
                        'Low_24h': stats['low_price'],
                        'Volume_24h': stats['volume']
                    })
                    
                import time
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"⚠️  Error getting data for {symbol}: {e}")
                continue
        
        if overview_data:
            df = pd.DataFrame(overview_data)
            print("✅ Market overview ready")
            return df
        else:
            print("❌ No market data available")
            return pd.DataFrame()


def demo_data_downloader():
    """
    Demo function showing how to use UnifiedDataDownloader
    """
    print("=== 統一數據下載器演示 ===")
    
    # Initialize downloader
    downloader = UnifiedDataDownloader()
    
    # Example 1: Download single pair
    print("\n1. 下載單一交易對 (BTC):")
    btc_data = downloader.download_single_pair(
        symbol='BTCUSDT',
        exchange='binance',
        interval='1h',
        days=7,
        market_type='spot'
    )
    
    if not btc_data.empty:
        summary = downloader.get_data_summary(btc_data, 'BTCUSDT')
        print(f"   數據點數: {summary['data_points']}")
        print(f"   價格範圍: ${summary['price_range']['min']:.2f} - ${summary['price_range']['max']:.2f}")
    
    # Example 2: Download multiple pairs
    print("\n2. 下載多個交易對:")
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    multi_data = downloader.download_multiple_pairs(
        symbols=symbols,
        interval='1h',
        days=3
    )
    
    print(f"   成功下載: {len(multi_data)} 個交易對")
    
    # Example 3: Market overview
    print("\n3. 市場概覽:")
    market_overview = downloader.get_current_market_overview()
    if not market_overview.empty:
        print(market_overview.round(2))
    
    # Example 4: Prepare ARIMA dataset
    print("\n4. 準備 ARIMA 數據集:")
    arima_data = downloader.create_dataset_for_arima(
        symbol='BTCUSDT',
        days=14,
        interval='1h'
    )
    
    return downloader, btc_data, multi_data


if __name__ == "__main__":
    downloader, btc_data, multi_data = demo_data_downloader()
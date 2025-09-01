"""
Data Download Script for Cryptocurrency Market Data

This script provides a command-line interface for downloading cryptocurrency
market data from various exchanges for quantitative analysis.

Usage:
    python scripts/download_data.py --symbol BTCUSDT --days 30 --interval 1h
    python scripts/download_data.py --popular --days 7
    python scripts/download_data.py --multiple BTCUSDT,ETHUSDT,BNBUSDT --days 14

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import sys
import os
import argparse
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection.data_downloader import UnifiedDataDownloader

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Download cryptocurrency market data for quantitative analysis'
    )
    
    # Symbol options
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        '--symbol', '-s',
        type=str,
        help='Single trading pair symbol (e.g., BTCUSDT)'
    )
    symbol_group.add_argument(
        '--multiple', '-m',
        type=str,
        help='Comma-separated list of symbols (e.g., BTCUSDT,ETHUSDT,BNBUSDT)'
    )
    symbol_group.add_argument(
        '--popular', '-p',
        action='store_true',
        help='Download popular trading pairs'
    )
    
    # Data parameters
    parser.add_argument(
        '--exchange', '-e',
        type=str,
        default='binance',
        choices=['binance', 'bybit'],
        help='Exchange to download from (default: binance)'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=str,
        default='1h',
        choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        help='Time interval for data (default: 1h)'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=30,
        help='Number of days of historical data (default: 30)'
    )
    
    parser.add_argument(
        '--market-type',
        type=str,
        default='spot',
        choices=['spot', 'futures'],
        help='Market type (default: spot)'
    )
    
    # Output options
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Don\'t save data to CSV files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for data files (default: data/raw)'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show detailed summary statistics'
    )
    
    parser.add_argument(
        '--market-overview',
        action='store_true',
        help='Show current market overview'
    )
    
    parser.add_argument(
        '--verify-ssl',
        action='store_true',
        help='Enable SSL certificate verification (default: disabled to avoid SSL issues)'
    )
    
    return parser.parse_args()

def download_single_symbol(downloader, symbol, args):
    """Download data for a single symbol"""
    print(f"\n{'='*50}")
    print(f"下載 {symbol} 數據")
    print(f"{'='*50}")
    
    data = downloader.download_single_pair(
        symbol=symbol,
        exchange=args.exchange,
        interval=args.interval,
        days=args.days,
        market_type=args.market_type,
        save_to_file=not args.no_save
    )
    
    if not data.empty and args.summary:
        summary = downloader.get_data_summary(data, symbol)
        print_summary(summary)
    
    return data

def download_multiple_symbols(downloader, symbols, args):
    """Download data for multiple symbols"""
    print(f"\n{'='*50}")
    print(f"下載多個交易對數據: {', '.join(symbols)}")
    print(f"{'='*50}")
    
    data_dict = downloader.download_multiple_pairs(
        symbols=symbols,
        exchange=args.exchange,
        interval=args.interval,
        days=args.days,
        market_type=args.market_type,
        save_to_file=not args.no_save
    )
    
    if args.summary:
        print(f"\n📊 多交易對數據摘要:")
        for symbol, data in data_dict.items():
            summary = downloader.get_data_summary(data, symbol)
            print(f"\n{symbol}:")
            print_summary(summary, indent=2)
    
    return data_dict

def print_summary(summary, indent=0):
    """Print data summary with proper formatting"""
    if not summary:
        return
    
    prefix = "  " * indent
    print(f"{prefix}📊 數據摘要:")
    print(f"{prefix}   期間: {summary['period']}")
    print(f"{prefix}   數據點數: {summary['data_points']:,}")
    print(f"{prefix}   價格範圍: ${summary['price_range']['min']:,.2f} - ${summary['price_range']['max']:,.2f}")
    print(f"{prefix}   當前價格: ${summary['price_range']['current']:,.2f}")
    print(f"{prefix}   平均收益率: {summary['returns']['mean']*100:+.4f}%")
    print(f"{prefix}   收益率波動: {summary['returns']['std']*100:.4f}%")
    print(f"{prefix}   平均成交量: {summary['volume']['mean']:,.2f}")

def main():
    """Main function"""
    args = parse_arguments()
    
    print("🚀 加密貨幣數據下載器")
    print(f"交易所: {args.exchange.upper()}")
    print(f"時間間隔: {args.interval}")
    print(f"歷史天數: {args.days}")
    print(f"市場類型: {args.market_type}")
    
    # Initialize data downloader
    downloader = UnifiedDataDownloader(data_folder=args.output_dir, verify_ssl=args.verify_ssl)
    
    # Show market overview if requested
    if args.market_overview:
        print(f"\n📈 當前市場概覽:")
        overview = downloader.get_current_market_overview()
        if not overview.empty:
            print(overview.to_string(index=False))
        else:
            print("無法獲取市場概覽")
    
    # Download data based on options
    data_results = {}
    
    if args.symbol:
        # Single symbol
        data = download_single_symbol(downloader, args.symbol, args)
        data_results[args.symbol] = data
        
    elif args.multiple:
        # Multiple symbols
        symbols = [s.strip().upper() for s in args.multiple.split(',')]
        data_results = download_multiple_symbols(downloader, symbols, args)
        
    elif args.popular:
        # Popular pairs
        print(f"\n{'='*50}")
        print(f"下載熱門交易對數據")
        print(f"{'='*50}")
        
        data_results = downloader.download_popular_pairs(
            exchange=args.exchange,
            interval=args.interval,
            days=args.days,
            market_type=args.market_type
        )
        
        if args.summary:
            print(f"\n📊 熱門交易對數據摘要:")
            for symbol, data in data_results.items():
                summary = downloader.get_data_summary(data, symbol)
                print(f"\n{symbol}:")
                print_summary(summary, indent=1)
    
    # Final summary
    successful_downloads = len([d for d in data_results.values() if not d.empty])
    total_attempts = len(data_results)
    total_data_points = sum(len(d) for d in data_results.values() if not d.empty)
    
    print(f"\n{'='*50}")
    print(f"下載完成！")
    print(f"{'='*50}")
    print(f"✅ 成功下載: {successful_downloads}/{total_attempts} 個交易對")
    print(f"📊 總數據點數: {total_data_points:,}")
    
    if not args.no_save:
        print(f"💾 數據已保存至: {args.output_dir}/{args.exchange}/")
    
    # Additional analysis suggestions
    if successful_downloads > 0:
        print(f"\n💡 接下來可以:")
        print(f"   1. 使用 ARIMA 模型分析: python scripts/demo_arima_btc.py")
        print(f"   2. 打開 Jupyter: jupyter notebook notebooks/stage1_foundations/01_arima_analysis.ipynb")
        print(f"   3. 查看結果文件夾: results/")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️  用戶取消下載")
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
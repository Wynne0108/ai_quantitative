"""
Multiple Timeframes Cointegration Testing

This script specifically tests cointegration analysis across different data timeframes:
- 15min, 1h, 4h, 1d intervals
- Same period length but different granularity
- Compare cointegration detection across timeframes

Usage: python scripts/test_multiple_timeframes.py

Author: Quantitative Trading Team
Date: 2025-09-02
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_collection.binance_api import BinanceDataCollector
from models.intermediate.cointegration import CointegrationAnalyzer


def collect_multiple_timeframe_data():
    """Collect data at different timeframes for the same period"""
    print("=== Collecting Multi-Timeframe Data ===")
    
    collector = BinanceDataCollector()
    
    # Define timeframes to test
    timeframes = {
        '15m': {'interval': '15m', 'days': 30, 'description': '15-minute candles, 30 days'},
        '1h': {'interval': '1h', 'days': 60, 'description': '1-hour candles, 60 days'},  
        '4h': {'interval': '4h', 'days': 120, 'description': '4-hour candles, 120 days'},
        '1d': {'interval': '1d', 'days': 365, 'description': '1-day candles, 365 days'}
    }
    
    # Symbols to collect
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
    
    collected_data = {}
    
    for timeframe_name, config in timeframes.items():
        print(f"\n[+] Collecting {timeframe_name} data ({config['description']})")
        
        timeframe_data = {}
        successful_symbols = 0
        
        for symbol in symbols:
            try:
                print(f"  Fetching {symbol} {config['interval']} data...")
                
                # Calculate start_time based on days back
                end_time = datetime.now()
                start_time = end_time - timedelta(days=config['days'])
                
                data = collector.get_historical_klines(
                    symbol=symbol,
                    interval=config['interval'],
                    start_time=start_time,
                    end_time=end_time,
                    limit=1000
                )
                
                if data is not None and len(data) > 100:
                    # Clean symbol name and use close prices
                    clean_symbol = symbol.replace('USDT', '')
                    timeframe_data[clean_symbol] = data['close']
                    successful_symbols += 1
                    print(f"    [+] {symbol}: {len(data)} data points")
                else:
                    print(f"    [-] {symbol}: Insufficient data")
                    
            except Exception as e:
                print(f"    [-] {symbol}: Failed - {e}")
                continue
        
        if successful_symbols >= 3:  # Need at least 3 assets for meaningful analysis
            # Align all data to same time index
            df = pd.DataFrame(timeframe_data)
            df = df.dropna()
            
            if len(df) > 50:  # Minimum data requirement
                collected_data[timeframe_name] = {
                    'data': df,
                    'config': config,
                    'symbols_collected': successful_symbols,
                    'data_points': len(df),
                    'date_range': (df.index[0], df.index[-1])
                }
                print(f"  [OK] {timeframe_name} dataset ready: {df.shape}")
            else:
                print(f"  [X] {timeframe_name}: Not enough aligned data ({len(df)} points)")
        else:
            print(f"  [X] {timeframe_name}: Not enough symbols collected ({successful_symbols}/5)")
    
    print(f"\n[Summary] Successfully collected {len(collected_data)} timeframes")
    return collected_data


def analyze_timeframe_cointegration(timeframe_data: dict, timeframe_name: str):
    """Analyze cointegration for a specific timeframe"""
    print(f"\n[Analysis] Analyzing {timeframe_name} timeframe...")
    
    data = timeframe_data['data']
    config = timeframe_data['config']
    
    print(f"  Data: {data.shape}")
    print(f"  Period: {timeframe_data['date_range'][0]} to {timeframe_data['date_range'][1]}")
    print(f"  Interval: {config['interval']}")
    
    results = {}
    
    # Test multiple parameter configurations for this timeframe
    test_configs = [
        {'lookback': min(252, len(data)//4), 'confidence': 0.05, 'name': 'Standard'},
        {'lookback': min(120, len(data)//6), 'confidence': 0.10, 'name': 'Relaxed'},
        {'lookback': min(60, len(data)//8), 'confidence': 0.15, 'name': 'Very Relaxed'},
        {'lookback': min(30, len(data)//10), 'confidence': 0.10, 'name': 'Short-term'}
    ]
    
    for test_config in test_configs:
        config_name = test_config['name']
        print(f"\n    Testing {config_name} configuration...")
        
        try:
            # Initialize analyzer
            analyzer = CointegrationAnalyzer(
                price_data=data,
                lookback_period=test_config['lookback'],
                confidence_level=test_config['confidence']
            )
            
            # Run cointegration analysis
            pairs_results = analyzer.analyze_pairs_cointegration()
            
            # Calculate summary metrics
            total_pairs = len(pairs_results)
            cointegrated_pairs = [(k, v) for k, v in pairs_results.items() if v['is_cointegrated']]
            cointegrated_count = len(cointegrated_pairs)
            
            config_results = {
                'analyzer': analyzer,
                'pairs_results': pairs_results,
                'total_pairs': total_pairs,
                'cointegrated_count': cointegrated_count,
                'detection_rate': cointegrated_count / total_pairs if total_pairs > 0 else 0,
                'best_p_value': min([p['min_p_value'] for p in pairs_results.values()]) if pairs_results else 1.0,
                'avg_r_squared': np.mean([p[f'test_{p["best_direction"]}']['r_squared'] 
                                        for p in pairs_results.values()]) if pairs_results else 0,
                'cointegrated_pairs': cointegrated_pairs
            }
            
            results[config_name] = config_results
            
            print(f"      Pairs tested: {total_pairs}")
            print(f"      Cointegrated: {cointegrated_count} ({config_results['detection_rate']:.1%})")
            print(f"      Best p-value: {config_results['best_p_value']:.6f}")
            print(f"      Avg R²: {config_results['avg_r_squared']:.4f}")
            
            # Test signal generation on best pair if exists
            if cointegrated_pairs:
                best_pair_key = cointegrated_pairs[0][0]
                print(f"      Testing signals on: {best_pair_key}")
                
                try:
                    signals = analyzer.generate_trading_signals(
                        pair_key=best_pair_key,
                        dynamic_threshold=True,
                        lookback_zscore=min(60, test_config['lookback']//2)
                    )
                    
                    config_results['signals'] = {
                        'generated': (signals['entry_signal'] != 0).sum(),
                        'valid_zscores': signals['zscore'].notna().sum(),
                        'stop_losses': (signals['stop_loss_signal'] != 0).sum() if 'stop_loss_signal' in signals else 0,
                        'signals_data': signals
                    }
                    
                    print(f"      Signals: {config_results['signals']['generated']}")
                    print(f"      Valid z-scores: {config_results['signals']['valid_zscores']}")
                    
                except Exception as e:
                    print(f"      Signal generation failed: {e}")
                    config_results['signals'] = None
            
        except Exception as e:
            print(f"    ❌ {config_name} failed: {e}")
            continue
    
    return results


def create_timeframe_comparison_report(all_results: dict):
    """Create comprehensive comparison report across timeframes"""
    print(f"\n[Report] Creating Timeframe Comparison Report...")
    
    # Prepare summary data
    summary_data = []
    
    for timeframe_name, timeframe_results in all_results.items():
        if 'analysis_results' not in timeframe_results:
            continue
            
        analysis_results = timeframe_results['analysis_results']
        data_info = timeframe_results['data_info']
        
        for config_name, config_results in analysis_results.items():
            summary_data.append({
                'timeframe': timeframe_name,
                'interval': data_info['config']['interval'],
                'data_points': data_info['data_points'],
                'config': config_name,
                'detection_rate': config_results['detection_rate'],
                'cointegrated_pairs': config_results['cointegrated_count'],
                'total_pairs': config_results['total_pairs'],
                'best_p_value': config_results['best_p_value'],
                'avg_r_squared': config_results['avg_r_squared'],
                'signals_generated': config_results.get('signals', {}).get('generated', 0) if config_results.get('signals') is not None else 0
            })
    
    # Create DataFrame for analysis
    summary_df = pd.DataFrame(summary_data)
    
    if summary_df.empty:
        print("No results to analyze")
        return None
    
    # Create report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MULTIPLE TIMEFRAMES COINTEGRATION ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Best performing configurations by timeframe
    report_lines.append("BEST PERFORMING CONFIGURATIONS BY TIMEFRAME")
    report_lines.append("-" * 50)
    
    for timeframe in summary_df['timeframe'].unique():
        tf_data = summary_df[summary_df['timeframe'] == timeframe]
        best_config = tf_data.loc[tf_data['detection_rate'].idxmax()]
        
        report_lines.append(f"\n{timeframe.upper()} ({best_config['interval']}):")
        report_lines.append(f"  Data Points: {best_config['data_points']}")
        report_lines.append(f"  Best Config: {best_config['config']}")
        report_lines.append(f"  Detection Rate: {best_config['detection_rate']:.1%}")
        report_lines.append(f"  Best p-value: {best_config['best_p_value']:.6f}")
        report_lines.append(f"  Avg R²: {best_config['avg_r_squared']:.4f}")
        report_lines.append(f"  Signals Generated: {best_config['signals_generated']}")
    
    # Overall rankings
    report_lines.append(f"\n\nOVERALL RANKINGS")
    report_lines.append("-" * 30)
    
    # Best detection rates
    best_detection = summary_df.nlargest(3, 'detection_rate')
    report_lines.append(f"\nTop 3 Detection Rates:")
    for idx, row in best_detection.iterrows():
        report_lines.append(f"  {row['timeframe']} ({row['config']}): {row['detection_rate']:.1%}")
    
    # Best statistical significance
    best_significance = summary_df.nsmallest(3, 'best_p_value')
    report_lines.append(f"\nTop 3 Statistical Significance:")
    for idx, row in best_significance.iterrows():
        report_lines.append(f"  {row['timeframe']} ({row['config']}): p={row['best_p_value']:.6f}")
    
    # Most active for trading
    best_signals = summary_df.nlargest(3, 'signals_generated')
    report_lines.append(f"\nTop 3 Trading Activity:")
    for idx, row in best_signals.iterrows():
        report_lines.append(f"  {row['timeframe']} ({row['config']}): {row['signals_generated']} signals")
    
    # Recommendations
    report_lines.append(f"\n\nRECOMMENDATIONS")
    report_lines.append("-" * 20)
    
    if summary_df['detection_rate'].max() > 0:
        best_overall = summary_df.loc[summary_df['detection_rate'].idxmax()]
        report_lines.append(f"Recommended Timeframe: {best_overall['timeframe']} ({best_overall['interval']})")
        report_lines.append(f"Recommended Configuration: {best_overall['config']}")
        report_lines.append(f"Expected Detection Rate: {best_overall['detection_rate']:.1%}")
    else:
        report_lines.append("No significant cointegration detected across any timeframe.")
        report_lines.append("Recommendations:")
        report_lines.append("1. Collect longer historical data (1-2 years)")
        report_lines.append("2. Try different asset combinations")
        report_lines.append("3. Consider alternative strategies (momentum, mean reversion)")
    
    report_lines.append("\n" + "=" * 80)
    
    # Save report
    report_text = "\n".join(report_lines)
    
    with open('results/model_outputs/multiple_timeframes_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Save summary data
    summary_df.to_csv('results/model_outputs/timeframes_summary.csv', index=False)
    
    print("[+] Report saved: results/model_outputs/multiple_timeframes_report.txt")
    print("[+] Summary data: results/model_outputs/timeframes_summary.csv")
    
    return summary_df


def create_timeframe_visualizations(all_results: dict, summary_df: pd.DataFrame):
    """Create visualizations comparing different timeframes"""
    print(f"\n[Viz] Creating visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multiple Timeframes Cointegration Analysis', fontsize=16)
        
        # Plot 1: Detection rates by timeframe
        if not summary_df.empty:
            timeframe_detection = summary_df.groupby('timeframe')['detection_rate'].max()
            
            axes[0, 0].bar(timeframe_detection.index, timeframe_detection.values * 100, alpha=0.7)
            axes[0, 0].set_title('Best Detection Rate by Timeframe')
            axes[0, 0].set_ylabel('Detection Rate (%)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for i, (timeframe, rate) in enumerate(timeframe_detection.items()):
                axes[0, 0].text(i, rate * 100 + 1, f'{rate:.1%}', ha='center')
        
        # Plot 2: p-values comparison
        if not summary_df.empty:
            timeframe_pvalues = summary_df.groupby('timeframe')['best_p_value'].min()
            
            axes[0, 1].bar(timeframe_pvalues.index, timeframe_pvalues.values, alpha=0.7, color='orange')
            axes[0, 1].set_title('Best p-values by Timeframe')
            axes[0, 1].set_ylabel('p-value')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5% threshold')
            axes[0, 1].legend()
        
        # Plot 3: Data points vs Detection rate
        if not summary_df.empty:
            axes[1, 0].scatter(summary_df['data_points'], summary_df['detection_rate'] * 100, 
                             c=summary_df['timeframe'].astype('category').cat.codes, alpha=0.7)
            axes[1, 0].set_xlabel('Data Points')
            axes[1, 0].set_ylabel('Detection Rate (%)')
            axes[1, 0].set_title('Data Points vs Detection Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Signal generation comparison
        if not summary_df.empty and summary_df['signals_generated'].sum() > 0:
            signal_data = summary_df[summary_df['signals_generated'] > 0]
            if not signal_data.empty:
                axes[1, 1].barh(range(len(signal_data)), signal_data['signals_generated'], alpha=0.7, color='green')
                axes[1, 1].set_yticks(range(len(signal_data)))
                axes[1, 1].set_yticklabels([f"{row['timeframe']}-{row['config']}" for _, row in signal_data.iterrows()])
                axes[1, 1].set_xlabel('Signals Generated')
                axes[1, 1].set_title('Trading Signal Generation')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No signals generated\nacross timeframes', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'No signals generated\nacross timeframes', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_title('Trading Signal Generation')
        
        plt.tight_layout()
        plt.savefig('results/plots/multiple_timeframes_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("[+] Visualization saved: results/plots/multiple_timeframes_comparison.png")
        
    except Exception as e:
        print(f"Visualization failed: {e}")


def main():
    """Main function to test multiple timeframes"""
    print("=== MULTIPLE TIMEFRAMES COINTEGRATION TESTING ===")
    
    # Step 1: Collect data for different timeframes
    print("\n[1] Step 1: Data Collection")
    timeframe_datasets = collect_multiple_timeframe_data()
    
    if not timeframe_datasets:
        print("[X] No data collected. Exiting...")
        return None, None
    
    # Step 2: Analyze each timeframe
    print(f"\n[2] Step 2: Analysis ({len(timeframe_datasets)} timeframes)")
    all_results = {}
    
    for timeframe_name, timeframe_data in timeframe_datasets.items():
        try:
            analysis_results = analyze_timeframe_cointegration(timeframe_data, timeframe_name)
            
            all_results[timeframe_name] = {
                'data_info': timeframe_data,
                'analysis_results': analysis_results
            }
            
        except Exception as e:
            print(f"[X] Analysis failed for {timeframe_name}: {e}")
            continue
    
    if not all_results:
        print("[X] No successful analyses. Exiting...")
        return None, None
    
    # Step 3: Generate comparison report
    print(f"\n[3] Step 3: Comparison Report")
    summary_df = create_timeframe_comparison_report(all_results)
    
    # Step 4: Create visualizations
    if summary_df is not None:
        print(f"\n[4] Step 4: Visualizations")
        create_timeframe_visualizations(all_results, summary_df)
    
    print(f"\n" + "="*60)
    print("MULTIPLE TIMEFRAMES ANALYSIS COMPLETE")
    print("="*60)
    print("[Info] Check results/model_outputs/ for detailed reports")
    print("[Info] Check results/plots/ for visualizations")
    
    if summary_df is not None and not summary_df.empty:
        best_result = summary_df.loc[summary_df['detection_rate'].idxmax()]
        print(f"\n[BEST] BEST PERFORMING CONFIGURATION:")
        print(f"   Timeframe: {best_result['timeframe']} ({best_result['interval']})")
        print(f"   Configuration: {best_result['config']}")
        print(f"   Detection Rate: {best_result['detection_rate']:.1%}")
        print(f"   Best p-value: {best_result['best_p_value']:.6f}")
    else:
        print(f"\n[Warning] No significant cointegration found across any timeframe")
        print(f"   Consider collecting longer data or trying different asset pairs")
    
    print("="*60)
    
    return all_results, summary_df


if __name__ == "__main__":
    results, summary = main()
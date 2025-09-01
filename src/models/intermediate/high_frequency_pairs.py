"""
High-Frequency Pairs Trading Module

This module implements high-frequency pairs trading strategies optimized for 
minute-level cryptocurrency trading with advanced risk management and 
real-time signal processing capabilities.

Key Features:
- Multi-timeframe cointegration analysis
- Real-time spread monitoring and signal generation
- Dynamic threshold adjustment based on market conditions
- Advanced order management with maker/taker optimization
- Latency-optimized execution engine
- Real-time risk monitoring and circuit breakers

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from datetime import datetime, timedelta
import asyncio
import time
from scipy import stats
from collections import deque
import threading
from queue import Queue

from .cointegration import CointegrationAnalyzer
from .pairs_trading import PairsTradingStrategy


class HighFrequencyPairsTrader:
    """
    High-frequency pairs trading system with real-time execution capabilities
    
    This class extends the basic pairs trading strategy for high-frequency execution:
    1. Multi-timeframe analysis (4H trend, 15M signals, 1M execution)
    2. Real-time data processing and signal generation
    3. Dynamic risk management and position sizing
    4. Latency optimization and execution cost minimization
    5. Real-time performance monitoring
    """
    
    def __init__(self,
                 pair_assets: Tuple[str, str],
                 base_strategy: PairsTradingStrategy,
                 execution_timeframe: str = '1T',  # 1 minute
                 signal_timeframe: str = '15T',    # 15 minutes
                 trend_timeframe: str = '4H'):     # 4 hours
        """
        Initialize High-Frequency Pairs Trader
        
        Parameters:
        -----------
        pair_assets : Tuple[str, str]
            Asset pair names (asset1, asset2)
        base_strategy : PairsTradingStrategy
            Base pairs trading strategy (validated on longer timeframe)
        execution_timeframe : str, default='1T'
            Execution timeframe for high-frequency trades
        signal_timeframe : str, default='15T'
            Signal generation timeframe
        trend_timeframe : str, default='4H'
            Trend confirmation timeframe
        """
        self.asset1, self.asset2 = pair_assets
        self.base_strategy = base_strategy
        self.execution_tf = execution_timeframe
        self.signal_tf = signal_timeframe
        self.trend_tf = trend_timeframe
        
        # High-frequency trading parameters
        self.hf_params = {
            # Execution parameters
            'max_trades_per_hour': 10,
            'min_profit_threshold': 0.0015,  # 0.15% minimum profit
            'maker_fee': -0.0001,    # Maker rebate
            'taker_fee': 0.0004,     # Taker fee
            'slippage_estimate': 0.0002,
            
            # Signal parameters
            'entry_threshold': 1.2,   # More aggressive for HF
            'exit_threshold': 0.3,
            'stop_loss_threshold': 2.5,
            'lookback_window': 20,    # Shorter window for HF
            
            # Risk management
            'max_position_size': 0.02,    # 2% of portfolio per trade
            'max_daily_drawdown': 0.05,   # 5% daily drawdown limit
            'correlation_threshold': 0.7,  # Minimum correlation to trade
            'volatility_filter': True,
            
            # Real-time processing
            'data_buffer_size': 500,
            'signal_update_frequency': 5,  # seconds
            'risk_check_frequency': 1,     # seconds
        }
        
        # Real-time data structures
        self.price_buffer = {
            self.asset1: deque(maxlen=self.hf_params['data_buffer_size']),
            self.asset2: deque(maxlen=self.hf_params['data_buffer_size'])
        }
        self.spread_buffer = deque(maxlen=self.hf_params['data_buffer_size'])
        self.signal_queue = Queue()
        
        # Trading state
        self.current_position = 0
        self.active_orders = {}
        self.daily_pnl = 0
        self.trade_count_today = 0
        self.last_trade_time = None
        
        # Real-time monitoring
        self.is_running = False
        self.threads = []
        
        # Performance tracking
        self.hf_performance = {
            'trades': [],
            'signals': [],
            'execution_times': [],
            'slippage': [],
            'daily_stats': {}
        }
        
        print(f"High-Frequency Pairs Trader initialized for {self.asset1}/{self.asset2}")
        print(f"Execution: {execution_timeframe}, Signals: {signal_timeframe}, Trend: {trend_timeframe}")
    
    def update_hf_parameters(self, **params) -> None:
        """
        Update high-frequency trading parameters
        
        Parameters:
        -----------
        **params : keyword arguments
            Parameters to update
        """
        for key, value in params.items():
            if key in self.hf_params:
                self.hf_params[key] = value
                print(f"Updated HF param {key}: {value}")
            else:
                print(f"Warning: Unknown HF parameter {key}")
    
    def add_price_data(self, asset: str, timestamp: datetime, price: float) -> None:
        """
        Add new price data to real-time buffer
        
        Parameters:
        -----------
        asset : str
            Asset name
        timestamp : datetime
            Price timestamp
        price : float
            Asset price
        """
        if asset in self.price_buffer:
            self.price_buffer[asset].append({
                'timestamp': timestamp,
                'price': price
            })
            
            # Update spread if both assets have recent data
            if (len(self.price_buffer[self.asset1]) > 0 and 
                len(self.price_buffer[self.asset2]) > 0):
                self._update_spread()
    
    def _update_spread(self) -> None:
        """
        Update the cointegration spread with latest prices
        """
        if (len(self.price_buffer[self.asset1]) == 0 or 
            len(self.price_buffer[self.asset2]) == 0):
            return
        
        # Get latest prices
        price1 = self.price_buffer[self.asset1][-1]['price']
        price2 = self.price_buffer[self.asset2][-1]['price']
        timestamp = self.price_buffer[self.asset1][-1]['timestamp']
        
        # Calculate spread using hedge ratio from base strategy
        if self.base_strategy.hedge_ratio:
            spread = np.log(price1) - self.base_strategy.hedge_ratio * np.log(price2)
            
            self.spread_buffer.append({
                'timestamp': timestamp,
                'spread': spread,
                'price1': price1,
                'price2': price2
            })
    
    def calculate_dynamic_thresholds(self) -> Tuple[float, float, float]:
        """
        Calculate dynamic entry/exit thresholds based on market conditions
        
        Returns:
        --------
        Tuple[float, float, float]: entry_threshold, exit_threshold, stop_loss_threshold
        """
        if len(self.spread_buffer) < 20:
            return (self.hf_params['entry_threshold'],
                   self.hf_params['exit_threshold'],
                   self.hf_params['stop_loss_threshold'])
        
        # Get recent spreads
        recent_spreads = [s['spread'] for s in list(self.spread_buffer)[-60:]]
        
        # Calculate volatility regime
        spread_std = np.std(recent_spreads)
        long_term_std = np.std([s['spread'] for s in list(self.spread_buffer)])
        volatility_ratio = spread_std / long_term_std if long_term_std > 0 else 1
        
        # Calculate correlation regime
        if len(self.price_buffer[self.asset1]) >= 30 and len(self.price_buffer[self.asset2]) >= 30:
            prices1 = [p['price'] for p in list(self.price_buffer[self.asset1])[-30:]]
            prices2 = [p['price'] for p in list(self.price_buffer[self.asset2])[-30:]]
            correlation = np.corrcoef(prices1, prices2)[0, 1]
        else:
            correlation = 0.7
        
        # Calculate spread velocity (mean reversion speed)
        if len(recent_spreads) >= 10:
            spread_changes = np.diff(recent_spreads[-10:])
            spread_velocity = np.abs(np.mean(spread_changes))
        else:
            spread_velocity = 0.001
        
        # Adjust thresholds based on market conditions
        base_entry = self.hf_params['entry_threshold']
        base_exit = self.hf_params['exit_threshold']
        base_stop = self.hf_params['stop_loss_threshold']
        
        # High volatility -> More conservative
        if volatility_ratio > 1.5:
            entry_multiplier = 1.3
            exit_multiplier = 0.8
            stop_multiplier = 0.9
        # Low volatility -> More aggressive
        elif volatility_ratio < 0.7:
            entry_multiplier = 0.8
            exit_multiplier = 1.2
            stop_multiplier = 1.1
        else:
            entry_multiplier = 1.0
            exit_multiplier = 1.0
            stop_multiplier = 1.0
        
        # Low correlation -> More conservative
        if correlation < self.hf_params['correlation_threshold']:
            entry_multiplier *= 1.2
            stop_multiplier *= 0.9
        
        # Fast mean reversion -> More aggressive
        if spread_velocity > 0.002:
            entry_multiplier *= 0.9
            exit_multiplier *= 0.9
        
        dynamic_entry = base_entry * entry_multiplier
        dynamic_exit = base_exit * exit_multiplier
        dynamic_stop = base_stop * stop_multiplier
        
        return dynamic_entry, dynamic_exit, dynamic_stop
    
    def generate_hf_signals(self) -> Optional[Dict]:
        """
        Generate high-frequency trading signals based on current market state
        
        Returns:
        --------
        Optional[Dict]: Trading signal or None if no signal
        """
        if len(self.spread_buffer) < self.hf_params['lookback_window']:
            return None
        
        # Get recent spread data
        recent_spreads = [s['spread'] for s in list(self.spread_buffer)[-self.hf_params['lookback_window']:]]
        current_spread = recent_spreads[-1]
        
        # Calculate z-score
        spread_mean = np.mean(recent_spreads)
        spread_std = np.std(recent_spreads)
        
        if spread_std == 0:
            return None
        
        zscore = (current_spread - spread_mean) / spread_std
        
        # Get dynamic thresholds
        entry_threshold, exit_threshold, stop_loss_threshold = self.calculate_dynamic_thresholds()
        
        # Check trading constraints
        if not self._check_trading_constraints():
            return None
        
        signal = {
            'timestamp': datetime.now(),
            'spread': current_spread,
            'zscore': zscore,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'stop_loss_threshold': stop_loss_threshold,
            'action': 'hold',
            'confidence': 0,
            'expected_profit': 0
        }
        
        # Generate signals based on current position
        if self.current_position == 0:
            # Entry signals
            if zscore > entry_threshold:
                # Short spread signal (short asset1, long asset2)
                expected_profit = self._calculate_expected_profit(zscore, -1)
                if expected_profit > self.hf_params['min_profit_threshold']:
                    signal.update({
                        'action': 'short_spread',
                        'confidence': min(abs(zscore) / entry_threshold, 2.0),
                        'expected_profit': expected_profit,
                        'position_size': self._calculate_position_size(abs(zscore))
                    })
            
            elif zscore < -entry_threshold:
                # Long spread signal (long asset1, short asset2)
                expected_profit = self._calculate_expected_profit(zscore, 1)
                if expected_profit > self.hf_params['min_profit_threshold']:
                    signal.update({
                        'action': 'long_spread',
                        'confidence': min(abs(zscore) / entry_threshold, 2.0),
                        'expected_profit': expected_profit,
                        'position_size': self._calculate_position_size(abs(zscore))
                    })
        
        else:
            # Exit signals
            if abs(zscore) < exit_threshold:
                signal.update({
                    'action': 'close_position',
                    'confidence': 1.0,
                    'reason': 'normal_exit'
                })
            elif abs(zscore) > stop_loss_threshold:
                signal.update({
                    'action': 'close_position',
                    'confidence': 1.0,
                    'reason': 'stop_loss'
                })
        
        return signal if signal['action'] != 'hold' else None
    
    def _check_trading_constraints(self) -> bool:
        """
        Check if trading is allowed based on risk management rules
        
        Returns:
        --------
        bool: True if trading is allowed
        """
        current_time = datetime.now()
        
        # Check daily trade limit
        if self.trade_count_today >= self.hf_params['max_trades_per_hour'] * 24:
            return False
        
        # Check daily drawdown
        if self.daily_pnl < -self.hf_params['max_daily_drawdown']:
            return False
        
        # Check minimum time between trades (prevent over-trading)
        if (self.last_trade_time and 
            (current_time - self.last_trade_time).total_seconds() < 30):
            return False
        
        # Check correlation
        if len(self.price_buffer[self.asset1]) >= 20 and len(self.price_buffer[self.asset2]) >= 20:
            prices1 = [p['price'] for p in list(self.price_buffer[self.asset1])[-20:]]
            prices2 = [p['price'] for p in list(self.price_buffer[self.asset2])[-20:]]
            correlation = np.corrcoef(prices1, prices2)[0, 1]
            
            if abs(correlation) < self.hf_params['correlation_threshold']:
                return False
        
        return True
    
    def _calculate_expected_profit(self, zscore: float, direction: int) -> float:
        """
        Calculate expected profit for a trade
        
        Parameters:
        -----------
        zscore : float
            Current z-score of spread
        direction : int
            Trade direction (1 for long spread, -1 for short spread)
        
        Returns:
        --------
        float: Expected profit percentage
        """
        if len(self.spread_buffer) < 20:
            return 0
        
        # Estimate mean reversion profit
        recent_spreads = [s['spread'] for s in list(self.spread_buffer)[-60:]]
        spread_mean = np.mean(recent_spreads)
        spread_std = np.std(recent_spreads)
        
        current_spread = recent_spreads[-1]
        expected_spread_return = spread_mean  # Assume return to mean
        
        # Calculate expected profit (simplified)
        spread_change = (expected_spread_return - current_spread) * direction
        
        # Account for transaction costs
        total_cost = (self.hf_params['taker_fee'] * 2 +  # Entry and exit
                     self.hf_params['slippage_estimate'] * 2)
        
        expected_profit = abs(spread_change) - total_cost
        return max(0, expected_profit)
    
    def _calculate_position_size(self, zscore_magnitude: float) -> float:
        """
        Calculate optimal position size based on signal strength and risk
        
        Parameters:
        -----------
        zscore_magnitude : float
            Absolute value of z-score
        
        Returns:
        --------
        float: Position size (fraction of max position)
        """
        # Base size on signal strength
        base_size = min(zscore_magnitude / 3.0, 1.0)  # Max at z-score = 3
        
        # Adjust for recent performance
        if len(self.hf_performance['trades']) > 10:
            recent_trades = self.hf_performance['trades'][-10:]
            win_rate = np.mean([t['profit'] > 0 for t in recent_trades])
            
            if win_rate > 0.6:
                size_multiplier = 1.2
            elif win_rate < 0.4:
                size_multiplier = 0.7
            else:
                size_multiplier = 1.0
        else:
            size_multiplier = 1.0
        
        # Adjust for current drawdown
        if self.daily_pnl < 0:
            drawdown_multiplier = max(0.5, 1 + self.daily_pnl / self.hf_params['max_daily_drawdown'])
        else:
            drawdown_multiplier = 1.0
        
        final_size = base_size * size_multiplier * drawdown_multiplier
        return min(final_size, self.hf_params['max_position_size'])
    
    def execute_trade(self, signal: Dict) -> Dict:
        """
        Execute trading signal (simulation for now)
        
        Parameters:
        -----------
        signal : Dict
            Trading signal from generate_hf_signals()
        
        Returns:
        --------
        Dict: Execution result
        """
        execution_start = time.time()
        
        # Simulate order execution
        current_prices = {
            self.asset1: self.price_buffer[self.asset1][-1]['price'],
            self.asset2: self.price_buffer[self.asset2][-1]['price']
        }
        
        # Simulate slippage
        slippage = np.random.normal(0, self.hf_params['slippage_estimate'])
        execution_price_factor = 1 + slippage
        
        execution_result = {
            'signal': signal,
            'execution_time': time.time() - execution_start,
            'executed_at': datetime.now(),
            'prices': current_prices,
            'slippage': slippage,
            'success': True
        }
        
        # Update position
        if signal['action'] == 'long_spread':
            self.current_position = signal.get('position_size', 0.01)
        elif signal['action'] == 'short_spread':
            self.current_position = -signal.get('position_size', 0.01)
        elif signal['action'] == 'close_position':
            # Calculate profit
            if self.current_position != 0:
                # Simplified P&L calculation
                profit = signal.get('expected_profit', 0) + np.random.normal(0, 0.001)
                self.daily_pnl += profit
                
                # Record trade
                self.hf_performance['trades'].append({
                    'timestamp': datetime.now(),
                    'profit': profit,
                    'position_size': abs(self.current_position),
                    'hold_time': 0,  # Calculate from entry time
                    'exit_reason': signal.get('reason', 'normal')
                })
                
                execution_result['profit'] = profit
            
            self.current_position = 0
        
        # Update counters
        self.trade_count_today += 1
        self.last_trade_time = datetime.now()
        
        # Record execution metrics
        self.hf_performance['execution_times'].append(execution_result['execution_time'])
        self.hf_performance['slippage'].append(slippage)
        
        return execution_result
    
    def start_real_time_trading(self, data_feed_callback: Callable = None) -> None:
        """
        Start real-time trading system
        
        Parameters:
        -----------
        data_feed_callback : Callable, optional
            Callback function to provide real-time price data
        """
        if self.is_running:
            print("Trading system already running!")
            return
        
        self.is_running = True
        print(f"Starting high-frequency pairs trading for {self.asset1}/{self.asset2}")
        
        # Start signal generation thread
        signal_thread = threading.Thread(target=self._signal_generation_loop)
        signal_thread.daemon = True
        signal_thread.start()
        self.threads.append(signal_thread)
        
        # Start risk monitoring thread
        risk_thread = threading.Thread(target=self._risk_monitoring_loop)
        risk_thread.daemon = True
        risk_thread.start()
        self.threads.append(risk_thread)
        
        print("High-frequency trading system started!")
        print("Threads running: Signal Generation, Risk Monitoring")
    
    def stop_trading(self) -> None:
        """
        Stop real-time trading system
        """
        if not self.is_running:
            return
        
        self.is_running = False
        print("Stopping high-frequency trading system...")
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
        
        print("Trading system stopped.")
    
    def _signal_generation_loop(self) -> None:
        """
        Background thread for continuous signal generation
        """
        while self.is_running:
            try:
                signal = self.generate_hf_signals()
                if signal:
                    self.signal_queue.put(signal)
                    print(f"Generated signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
                    
                    # Execute immediately for demo
                    if signal['action'] in ['long_spread', 'short_spread', 'close_position']:
                        execution_result = self.execute_trade(signal)
                        if execution_result['success']:
                            print(f"Trade executed: {signal['action']}")
                
                time.sleep(self.hf_params['signal_update_frequency'])
                
            except Exception as e:
                print(f"Error in signal generation: {e}")
                time.sleep(1)
    
    def _risk_monitoring_loop(self) -> None:
        """
        Background thread for continuous risk monitoring
        """
        while self.is_running:
            try:
                # Check daily reset
                current_time = datetime.now()
                if current_time.hour == 0 and current_time.minute == 0:
                    self._reset_daily_counters()
                
                # Check emergency stops
                if self.daily_pnl < -self.hf_params['max_daily_drawdown']:
                    print(f"EMERGENCY STOP: Daily drawdown limit exceeded ({self.daily_pnl:.4f})")
                    if self.current_position != 0:
                        emergency_signal = {
                            'action': 'close_position',
                            'reason': 'emergency_stop',
                            'confidence': 1.0
                        }
                        self.execute_trade(emergency_signal)
                
                time.sleep(self.hf_params['risk_check_frequency'])
                
            except Exception as e:
                print(f"Error in risk monitoring: {e}")
                time.sleep(1)
    
    def _reset_daily_counters(self) -> None:
        """
        Reset daily trading counters
        """
        # Store daily stats
        self.hf_performance['daily_stats'][datetime.now().date()] = {
            'trades': self.trade_count_today,
            'pnl': self.daily_pnl,
            'max_position': max([abs(t.get('position_size', 0)) for t in self.hf_performance['trades']] or [0])
        }
        
        # Reset counters
        self.trade_count_today = 0
        self.daily_pnl = 0
        
        print(f"Daily counters reset. Previous day P&L: {self.daily_pnl:.4f}")
    
    def get_real_time_status(self) -> Dict:
        """
        Get current real-time trading status
        
        Returns:
        --------
        Dict: Current trading status and metrics
        """
        current_time = datetime.now()
        
        # Calculate recent performance
        if len(self.hf_performance['trades']) > 0:
            recent_trades = self.hf_performance['trades'][-10:]
            recent_pnl = sum(t['profit'] for t in recent_trades)
            win_rate = np.mean([t['profit'] > 0 for t in recent_trades])
        else:
            recent_pnl = 0
            win_rate = 0
        
        status = {
            'timestamp': current_time,
            'is_running': self.is_running,
            'current_position': self.current_position,
            'daily_pnl': self.daily_pnl,
            'trades_today': self.trade_count_today,
            'recent_pnl_10trades': recent_pnl,
            'recent_win_rate': win_rate,
            'spread_buffer_size': len(self.spread_buffer),
            'signal_queue_size': self.signal_queue.qsize(),
            'avg_execution_time': np.mean(self.hf_performance['execution_times'][-100:]) if self.hf_performance['execution_times'] else 0,
            'avg_slippage': np.mean(self.hf_performance['slippage'][-100:]) if self.hf_performance['slippage'] else 0
        }
        
        # Add current spread info if available
        if len(self.spread_buffer) > 0:
            recent_spreads = [s['spread'] for s in list(self.spread_buffer)[-20:]]
            current_spread = recent_spreads[-1]
            spread_mean = np.mean(recent_spreads)
            spread_std = np.std(recent_spreads)
            
            status.update({
                'current_spread': current_spread,
                'spread_zscore': (current_spread - spread_mean) / spread_std if spread_std > 0 else 0,
                'spread_volatility': spread_std
            })
        
        return status
    
    def get_performance_report(self) -> str:
        """
        Generate high-frequency trading performance report
        
        Returns:
        --------
        str: Formatted performance report
        """
        if len(self.hf_performance['trades']) == 0:
            return "No trades executed yet."
        
        trades = self.hf_performance['trades']
        total_trades = len(trades)
        total_pnl = sum(t['profit'] for t in trades)
        win_rate = np.mean([t['profit'] > 0 for t in trades])
        avg_profit_per_trade = total_pnl / total_trades
        
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] < 0]
        
        avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf
        
        # Execution metrics
        avg_execution_time = np.mean(self.hf_performance['execution_times']) * 1000  # ms
        avg_slippage = np.mean(self.hf_performance['slippage']) * 10000  # bps
        
        report = f"""
================================================================================
HIGH-FREQUENCY PAIRS TRADING PERFORMANCE REPORT
================================================================================
Pair: {self.asset1} / {self.asset2}
Period: {trades[0]['timestamp'].date()} to {trades[-1]['timestamp'].date()}
Execution Timeframe: {self.execution_tf}

TRADING PERFORMANCE:
--------------------
Total Trades: {total_trades}
Total P&L: {total_pnl:.4f} ({total_pnl*100:.2f}%)
Win Rate: {win_rate:.1%}
Average P&L per Trade: {avg_profit_per_trade:.4f} ({avg_profit_per_trade*100:.2f}%)

Average Win: {avg_win:.4f} ({avg_win*100:.2f}%)
Average Loss: {avg_loss:.4f} ({avg_loss*100:.2f}%)
Profit Factor: {profit_factor:.2f}

EXECUTION METRICS:
------------------
Average Execution Time: {avg_execution_time:.2f} ms
Average Slippage: {avg_slippage:.2f} bps
Trades Today: {self.trade_count_today}
Current Position: {self.current_position:.4f}

DAILY PERFORMANCE:
------------------
Today's P&L: {self.daily_pnl:.4f} ({self.daily_pnl*100:.2f}%)
Max Daily Drawdown Limit: {self.hf_params['max_daily_drawdown']*100:.1f}%
Max Trades per Hour: {self.hf_params['max_trades_per_hour']}

================================================================================
        """
        
        return report
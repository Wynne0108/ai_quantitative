"""
Intermediate Models for Quantitative Trading - Stage 2

This module contains advanced models that build upon the foundation models:
- Cointegration Analysis: Multi-asset relationship modeling
- Pairs Trading Strategies: Statistical arbitrage
- Advanced Risk Management: Portfolio-level risk models

Author: Quantitative Trading Team
Date: 2025-01-01
"""

from .cointegration import CointegrationAnalyzer
from .pairs_trading import PairsTradingStrategy
from .statistical_tests import StatisticalTests

__all__ = [
    'CointegrationAnalyzer',
    'PairsTradingStrategy', 
    'StatisticalTests'
]
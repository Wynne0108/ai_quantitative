# 📊 多時間窗口協整分析使用指南

## 🎯 概述

我已經為你的協整分析系統添加了完整的多時間窗口測試功能。根據你的測試結果，可以看到：

- **當前數據質量**：1080個4小時數據點，約6個月數據
- **檢測挑戰**：在10%寬松閾值下仍無法檢測到穩定的協整關係
- **滾動穩定性**：25.1%的平均檢測率，標準差18.6%，顯示關係不穩定

## 🚀 如何使用新功能

### 1. 快速參數測試 (推薦)

```bash
# 快速測試5種不同參數配置
python scripts/quick_timeframe_test.py
```

這會測試：
- 短期配置 (60天回顧 + 30期z-score)
- 中期配置 (120天回顧 + 60期z-score)  
- 長期配置 (200天回顧 + 90期z-score)
- 激進配置 (閾值1.5)
- 保守配置 (閾值2.5)

### 2. 完整多時間框架分析

```bash
# 完整的多時間框架和參數組合測試
python scripts/test_multi_timeframe_cointegration.py
```

這會測試：
- 多個時間間隔 (15分鐘、1小時、4小時、1天)
- 各種參數組合
- 滾動穩定性分析

### 3. 在代碼中使用新功能

```python
from src.models.intermediate.cointegration import CointegrationAnalyzer

# 初始化分析器
analyzer = CointegrationAnalyzer(
    price_data=your_data,
    lookback_period=120,  # 根據測試結果選擇
    confidence_level=0.10  # 更寬鬆的閾值
)

# 基礎協整分析
pairs_results = analyzer.analyze_pairs_cointegration()

# 滾動穩定性測試
rolling_results = analyzer.rolling_cointegration_test(
    window_size=120,  # 滾動窗口大小
    step_size=20,     # 滾動步長
    min_periods=60    # 最小數據需求
)

# 可視化滾動結果
analyzer.plot_rolling_cointegration_analysis(rolling_results)

# 動態閾值信號生成
signals = analyzer.generate_trading_signals(
    pair_key='BTC_ETH',
    dynamic_threshold=True,  # 使用動態閾值
    lookback_zscore=60
)
```

## 📈 根據你的數據的建議

### 當前問題分析
1. **時間跨度不足**：6個月數據可能不夠捕捉長期協整關係
2. **市場環境**：可能處於高波動期，破壞了協整關係
3. **資產選擇**：主流加密貨幣可能相關性過高但缺乏穩定的協整

### 改進建議

**1. 數據改進**
```python
# 收集更長時間數據 (建議至少1-2年)
downloader = UnifiedDataDownloader()
longer_data = downloader.download_single_pair('BTCUSDT', days=730)  # 2年數據

# 測試不同資產組合
alternative_pairs = [
    'BTCUSDT', 'WBTC-USD',  # BTC vs 包裝BTC
    'ETHUSDT', 'stETH-USD', # ETH vs 質押ETH
    'USDTUSD', 'USDCUSD'   # 穩定幣對
]
```

**2. 參數調整**
```python
# 基於滾動分析結果，建議使用：
analyzer = CointegrationAnalyzer(
    price_data=data,
    lookback_period=200,     # 使用較長回顧期
    confidence_level=0.15    # 進一步放寬到15%
)

# 動態信號設置
signals = analyzer.generate_trading_signals(
    pair_key=best_pair,
    dynamic_threshold=True,
    lookback_zscore=90      # 更長的z-score計算窗口
)
```

**3. 替代策略**
```python
# 如果協整關係不穩定，考慮：

# 1. 短期均值回歸 (不依賴協整)
signals = analyzer.generate_trading_signals(
    pair_key=pair,
    entry_threshold=1.5,    # 更低閾值
    exit_threshold=0.2,     # 更快退出
    lookback_zscore=30      # 更短窗口
)

# 2. 條件協整 (只在特定市場條件下交易)
# 只在滾動檢測率 > 50% 的窗口進行交易
```

## 📊 結果解讀指南

### 滾動分析指標

- **平均檢測率 25.1%**：說明協整關係只在1/4的時間內存在
- **標準差 18.6%**：顯示關係非常不穩定
- **範圍 0-80%**：存在一些時期有強協整，但不持續

### 建議閾值設置

基於你的數據特征：
- **探索性分析**：confidence_level=0.15
- **信號生成**：entry_threshold=1.5, exit_threshold=0.3
- **風險管理**：max_loss_pct=0.015 (1.5%止損)

## 🛠️ 下一步行動計劃

1. **立即行動**：運行 `python scripts/quick_timeframe_test.py` 看結果
2. **數據擴展**：收集更長期歷史數據 (至少1年)
3. **資產探索**：測試更有可能協整的資產對
4. **策略調整**：考慮短期均值回歸而非長期協整

## 🔧 故障排除

**如果沒有檢測到協整關係**：
1. 降低 confidence_level 到 0.20
2. 減少 lookback_period 到 60
3. 嘗試不同的資產組合
4. 檢查數據質量和完整性

**如果信號生成失敗**：
1. 確保有至少一個協整對
2. 檢查 zscore 計算窗口不要太大
3. 驗證動態閾值計算

記住：協整分析在加密貨幣市場中具有挑戰性，因為這個市場相對年輕且波動性高。不要期望像傳統金融市場那樣穩定的關係。
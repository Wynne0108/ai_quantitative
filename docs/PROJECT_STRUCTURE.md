# Crypto Quantitative Trading - 專案結構

這是虛擬貨幣永續合約量化交易專案的完整目錄結構，按照學習路線圖分階段組織。

## 📁 專案目錄結構

```
ai_quantitative/
│
├── docs/                          # 文檔資料
│   ├── LEARNING.md                # 學習路線圖
│   └── PROJECT_STRUCTURE.md       # 專案結構說明
│
├── data/                          # 數據存放區
│   ├── raw/                      # 原始數據
│   │   ├── binance/              # Binance 數據
│   │   └── bybit/                # Bybit 數據
│   ├── processed/                # 處理後數據
│   └── cache/                    # 快取數據
│
├── src/                          # 核心程式碼
│   ├── data_collection/          # 數據收集模組
│   │   ├── __init__.py
│   │   ├── binance_api.py        # Binance API 接口
│   │   ├── bybit_api.py          # Bybit API 接口
│   │   └── data_downloader.py    # 統一數據下載器
│   │
│   ├── models/                   # 模型實作
│   │   ├── __init__.py
│   │   ├── foundations/          # 基礎模型 (階段一)
│   │   │   ├── __init__.py
│   │   │   ├── arima_model.py    # ARIMA 時間序列預測
│   │   │   ├── garch_model.py    # GARCH 波動率預測
│   │   │   └── gbm_model.py      # 幾何布朗運動模擬
│   │   │
│   │   ├── intermediate/         # 進階模型 (階段二)
│   │   │   ├── __init__.py
│   │   │   ├── cointegration.py  # 協整分析
│   │   │   ├── mean_reversion.py # 均值回歸策略
│   │   │   └── risk_models.py    # VaR/CVaR 風險模型
│   │   │
│   │   └── advanced/             # 前沿方法 (階段三)
│   │       ├── __init__.py
│   │       ├── ml_models.py      # 機器學習模型
│   │       ├── lstm_model.py     # LSTM 深度學習
│   │       └── transformer.py   # Transformer 模型
│   │
│   ├── strategies/               # 交易策略
│   │   ├── __init__.py
│   │   ├── arbitrage.py          # 套利策略
│   │   ├── momentum.py           # 動量策略
│   │   └── mean_reversion_strategy.py # 均值回歸策略
│   │
│   ├── backtesting/              # 回測框架
│   │   ├── __init__.py
│   │   ├── backtest_engine.py    # 回測引擎
│   │   └── performance_metrics.py # 績效指標
│   │
│   ├── risk_management/          # 風險管理
│   │   ├── __init__.py
│   │   ├── position_sizing.py    # 倉位管理
│   │   └── risk_calculator.py    # 風險計算
│   │
│   └── utils/                    # 工具函數
│       ├── __init__.py
│       ├── data_processing.py    # 數據處理工具
│       ├── technical_indicators.py # 技術指標
│       └── visualization.py     # 視覺化工具
│
├── notebooks/                    # Jupyter 筆記本
│   ├── stage1_foundations/       # 基礎模型實驗
│   │   ├── 01_arima_analysis.ipynb
│   │   ├── 02_garch_volatility.ipynb
│   │   └── 03_gbm_simulation.ipynb
│   │
│   ├── stage2_intermediate/      # 進階模型實驗
│   │   ├── 04_cointegration_analysis.ipynb
│   │   ├── 05_mean_reversion_strategy.ipynb
│   │   └── 06_risk_management.ipynb
│   │
│   └── stage3_advanced/          # 前沿方法實驗
│       ├── 07_ml_prediction.ipynb
│       ├── 08_lstm_forecast.ipynb
│       └── 09_transformer_analysis.ipynb
│
├── tests/                        # 測試程式碼
│   ├── __init__.py
│   ├── test_models/
│   ├── test_strategies/
│   └── test_data_collection/
│
├── scripts/                      # 執行腳本
│   ├── download_data.py          # 數據下載腳本
│   ├── run_backtest.py           # 回測執行腳本
│   └── model_training.py         # 模型訓練腳本
│
├── config/                       # 配置文件
│   ├── api_config.py             # API 配置
│   ├── model_config.py           # 模型參數配置
│   └── trading_config.py         # 交易參數配置
│
├── results/                      # 結果輸出
│   ├── backtest_results/         # 回測結果
│   ├── model_outputs/            # 模型輸出
│   └── plots/                    # 圖表輸出
│
├── requirements.txt              # Python 依賴套件
├── README.md                     # 專案說明
└── .gitignore                    # Git 忽略文件
```

## 🚀 學習階段對應

### 階段一：基礎模型
- **程式碼位置**: `src/models/foundations/`
- **實驗筆記**: `notebooks/stage1_foundations/`
- **重點模型**: ARIMA, GARCH, GBM

### 階段二：進階模型
- **程式碼位置**: `src/models/intermediate/`
- **實驗筆記**: `notebooks/stage2_intermediate/`
- **重點模型**: 協整分析, 均值回歸, VaR/CVaR

### 階段三：前沿方法
- **程式碼位置**: `src/models/advanced/`
- **實驗筆記**: `notebooks/stage3_advanced/`
- **重點模型**: ML, LSTM, Transformer

## 💡 使用建議

1. **開發流程**: 先在 notebooks 中實驗 → 確認效果後移至 src 模組化
2. **數據管理**: 原始數據放 `data/raw/`，處理後放 `data/processed/`
3. **版本控制**: 重要模型和策略都要有對應的測試案例
4. **結果保存**: 所有實驗結果統一存放在 `results/` 目錄

## 📦 核心套件依賴

```python
# 基礎套件
pandas
numpy
matplotlib
seaborn

# 時間序列與統計
statsmodels
arch
scipy

# 機器學習
scikit-learn
xgboost

# 深度學習
tensorflow
# 或 pytorch

# 數據獲取
requests
ccxt

# 其他工具
jupyter
pytest
```
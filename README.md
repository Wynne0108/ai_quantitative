# 🚀 Crypto Quantitative Trading - 虛擬貨幣量化交易專案

這是一個專為學習虛擬貨幣量化交易而設計的完整專案，按照循序漸進的學習路線圖實作各種量化模型與策略。

## 📋 專案概述

本專案分為三個學習階段：

- **🔰 階段一：基礎模型** - ARIMA、GARCH、GBM
- **📈 階段二：進階模型** - 協整分析、均值回歸、風險管理
- **🤖 階段三：前沿方法** - 機器學習、深度學習

## 🏗️ 專案結構

```
ai_quantitative/
├── 📚 docs/                    # 文檔資料
├── 📊 data/                    # 數據存放區  
├── 🔧 src/                     # 核心程式碼
│   ├── data_collection/        # 數據收集模組
│   ├── models/                 # 模型實作
│   │   ├── foundations/        # 基礎模型 (ARIMA, GARCH, GBM)
│   │   ├── intermediate/       # 進階模型
│   │   └── advanced/          # 前沿方法
│   └── utils/                 # 工具函數
├── 📓 notebooks/              # Jupyter 實驗筆記
├── 🧪 tests/                  # 測試程式碼
├── 📜 scripts/                # 執行腳本
└── 📈 results/               # 結果輸出
```

## 🚀 快速開始

### 1. 環境設置

```bash
# 克隆專案 (如果使用 git)
git clone <repository-url>
cd ai_quantitative

# 安裝依賴套件
pip install -r requirements.txt
```

### 2. 下載數據

```bash
# 下載 BTC 30天小時線數據
python scripts/download_data.py --symbol BTCUSDT --days 30 --interval 1h

# 下載熱門交易對數據  
python scripts/download_data.py --popular --days 7

# 下載多個交易對
python scripts/download_data.py --multiple BTCUSDT,ETHUSDT,BNBUSDT --days 14

# 查看市場概覽
python scripts/download_data.py --symbol BTCUSDT --days 1 --market-overview
```

### 3. 執行 ARIMA 分析

```bash
# 運行 BTC ARIMA 分析演示
python scripts/demo_arima_btc.py
```

### 4. 使用 Jupyter Notebook

```bash
# 啟動 Jupyter
jupyter notebook

# 打開 ARIMA 分析筆記本
# 瀏覽至: notebooks/stage1_foundations/01_arima_analysis.ipynb
```

## 📖 學習路線

### 🔰 階段一：基礎模型 (當前實作完成)

**已實作功能：**
- ✅ **ARIMA 模型** (`src/models/foundations/arima_model.py`)
  - 自動平穩性檢測 (ADF test)
  - 參數優選 (Grid Search + AIC/BIC)
  - 模型診斷與評估
  - 價格預測與信心區間
  - 完整視覺化分析

- ✅ **數據收集系統**
  - Binance API 整合
  - Bybit API 支援
  - 統一數據下載器
  - 多交易對批量下載

- ✅ **實驗筆記本**
  - 詳細的 ARIMA 教學筆記
  - 互動式數據分析
  - 視覺化圖表生成

**下一步：**
- 🔄 GARCH 波動率預測模型
- 🔄 GBM 價格模擬模型

### 📈 階段二：進階模型 (規劃中)

- 協整分析 (Cointegration)
- 均值回歸策略 (Mean Reversion)
- VaR/CVaR 風險管理

### 🤖 階段三：前沿方法 (規劃中)

- 機器學習模型 (XGBoost, Random Forest)
- 深度學習模型 (LSTM, Transformer)
- 特徵工程與模型集成

## 💡 使用範例

### ARIMA 模型分析

```python
from src.models.foundations.arima_model import ARIMAModel
from src.data_collection.data_downloader import UnifiedDataDownloader

# 下載數據
downloader = UnifiedDataDownloader()
btc_data = downloader.create_dataset_for_arima('BTCUSDT', days=30)

# ARIMA 分析
arima = ARIMAModel(btc_data, 'close')
arima.check_stationarity()
best_params, _ = arima.find_best_arima_params()
arima.fit_arima(order=best_params)
forecast = arima.forecast(steps=24)  # 24小時預測
arima.plot_forecast()
```

### 數據下載

```python
from src.data_collection.data_downloader import UnifiedDataDownloader

downloader = UnifiedDataDownloader()

# 單一交易對
btc_data = downloader.download_single_pair('BTCUSDT', days=30)

# 多個交易對
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
multi_data = downloader.download_multiple_pairs(symbols, days=7)

# 市場概覽
overview = downloader.get_current_market_overview()
```

## 📊 支援的交易所

- ✅ **Binance** - 現貨與期貨市場
- ✅ **Bybit** - 現貨與衍生品市場
- 🔄 更多交易所支援開發中...

## 📈 支援的時間間隔

- **分鐘線**: 1m, 3m, 5m, 15m, 30m
- **小時線**: 1h, 2h, 4h, 6h, 8h, 12h  
- **日線**: 1d, 3d
- **週月線**: 1w, 1M

## 🧪 測試

```bash
# 運行所有測試
pytest tests/

# 運行特定測試
pytest tests/test_models/
```

## 📁 輸出文件

執行分析後，結果會保存在以下位置：

- `results/plots/` - 圖表輸出
- `results/model_outputs/` - 模型預測結果
- `results/backtest_results/` - 回測結果
- `data/raw/` - 原始數據
- `data/processed/` - 處理後數據

## 🔧 配置

主要配置文件位於 `config/` 目錄：
- `api_config.py` - API 設置
- `model_config.py` - 模型參數
- `trading_config.py` - 交易參數

## 📚 學習資源

- 📖 [學習路線圖](docs/LEARNING.md)
- 🏗️ [專案結構說明](docs/PROJECT_STRUCTURE.md)
- 📓 [Jupyter 教學筆記](notebooks/stage1_foundations/)
- 📊 [模型文檔](src/models/)

## 🤝 貢獻

歡迎提交 Issue 或 Pull Request！

## ⚠️ 免責聲明

本專案僅供學習和研究目的。任何投資決策都應該基於您自己的分析和風險承受能力。作者不對任何投資損失承擔責任。

## 📄 授權

MIT License - 詳見 LICENSE 文件

---

**開始您的量化交易學習之旅！** 🎯

如有任何問題，請查看文檔或提交 Issue。
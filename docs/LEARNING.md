# Crypto Perpetual Futures Quantitative Trading - Learning Roadmap

這是一份針對 **虛擬貨幣永續合約交易** 的量化模型學習路線圖，分為基礎 → 進階 → 前沿三個階段。  
目標是透過數學、統計與程式實作，逐步掌握量化交易的核心模型與策略。

---

## 🚀 階段一：基礎模型 (Foundations)

### 學習內容
- **ARIMA (時間序列預測)**
- **GARCH (波動率預測)**
- **GBM (價格模擬 - 幾何布朗運動)**

### 工具
- Python: `pandas`, `numpy`, `statsmodels`, `arch`, `matplotlib`

### 小專案
- 下載 BTC 永續合約歷史 K 線資料  
- 使用 ARIMA 預測下一期價格  
- 使用 GARCH 預測未來一週波動率  
- 用 GBM 模擬 100 條可能的 BTC 價格路徑  

---

## 📈 階段二：進階模型 (Intermediate)

### 學習內容
- **協整 (Cointegration)** → 價差交易
- **均值回歸 (Mean Reversion, Ornstein-Uhlenbeck Process)**
- **VaR / CVaR (風險控管模型)**

### 工具
- Python: `statsmodels`, `scipy`, `numpy`, `matplotlib`

### 小專案
- 檢驗 BTC 永續 vs 現貨 價格是否協整  
- 建立簡單的「均值回歸」套利策略  
- 使用 VaR/CVaR 計算策略在 95% 信心水準下的風險  

---

## 🤖 階段三：前沿方法 (Advanced / Cutting-edge)

### 學習內容
- **機器學習 (ML)**: Regression, Random Forest, XGBoost
- **深度學習 (DL)**: LSTM, Transformer for time series

### 工具
- Python: `scikit-learn`, `xgboost`, `tensorflow` / `pytorch`

### 小專案
- 構建一個 ML 模型，使用技術指標 (RSI, MACD, VWAP) 預測下一根 K 線漲跌  
- 訓練一個 LSTM 模型，用 BTC 5 分鐘 K 線預測下一根收盤價  
- 嘗試 Transformer 模型，分析長期時間序列特徵  

---

## 📚 建議學習順序

1. **數學基礎**  
   - 機率分布、統計檢定、線性代數、微積分  
2. **程式基礎**  
   - Python 資料處理 (`pandas`, `numpy`)  
   - 畫圖 (`matplotlib`)  
3. **金融專用模型**  
   - ARIMA, GARCH, GBM → 模擬價格與波動  
   - 協整、均值回歸 → 找套利機會  
   - VaR/CVaR → 風險管理  
4. **機器學習/深度學習**  
   - 機器學習模型：特徵工程 + 預測  
   - 深度學習模型：LSTM / Transformer  

---

## ⏰ 學習規劃建議

- **第 1 個月** → 基礎模型 (ARIMA, GARCH, GBM)  
- **第 2 個月** → 進階模型 (協整、均值回歸、VaR)  
- **第 3 個月** → 前沿方法 (ML/DL)  

每階段都要搭配 **實際數據專案**，避免只學理論。  

---

## 🔗 資料來源

- [Binance API](https://binance-docs.github.io/apidocs/)  
- [Bybit API](https://bybit-exchange.github.io/docs/)  
- Python 套件文件: `statsmodels`, `arch`, `scikit-learn`, `pytorch`, `tensorflow`

---

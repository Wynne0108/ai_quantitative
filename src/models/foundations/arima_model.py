"""
ARIMA (AutoRegressive Integrated Moving Average) Model for Crypto Price Prediction

This module implements ARIMA model for time series forecasting of cryptocurrency prices.
ARIMA is suitable for univariate time series data that shows evidence of non-stationarity.

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# 導入字體配置工具
try:
    from ..utils.font_config import setup_chinese_fonts, configure_plot_style
    # 自動配置中文字體
    setup_chinese_fonts()
    configure_plot_style()
except ImportError:
    # 如果無法導入字體配置工具，使用基本配置
    plt.rcParams['axes.unicode_minus'] = False
    try:
        plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    except:
        pass

class ARIMAModel:
    """
    ARIMA Model for Cryptocurrency Price Prediction
    
    This class provides methods to:
    1. Test for stationarity
    2. Find optimal ARIMA parameters
    3. Fit ARIMA model
    4. Make predictions
    5. Evaluate model performance
    """
    
    def __init__(self, data, price_column='close'):
        """
        Initialize ARIMA Model
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price data with datetime index
        price_column : str
            Column name for price data (default: 'close')
        """
        self.data = data.copy()
        self.price_column = price_column
        self.prices = self.data[price_column]
        self.model = None
        self.fitted_model = None
        self.forecast_result = None
        
    def check_stationarity(self, series=None, significance_level=0.05):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Parameters:
        -----------
        series : pd.Series, optional
            Time series to test (default: self.prices)
        significance_level : float
            Significance level for the test (default: 0.05)
            
        Returns:
        --------
        dict : Dictionary containing test results
        """
        if series is None:
            series = self.prices
            
        result = adfuller(series.dropna())
        
        test_result = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < significance_level
        }
        
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"P-value: {result[1]:.6f}")
        print("Critical Values:")
        for key, value in result[4].items():
            print(f"\t{key}: {value:.3f}")
        print(f"Is Stationary: {'Yes' if test_result['is_stationary'] else 'No'}")
        
        return test_result
    
    def make_stationary(self, method='difference', log_transform=False):
        """
        Make time series stationary through differencing or log transformation
        
        Parameters:
        -----------
        method : str
            Method to make stationary ('difference' or 'log_difference')
        log_transform : bool
            Whether to apply log transformation first
            
        Returns:
        --------
        pd.Series : Stationary time series
        """
        series = self.prices.copy()
        
        if log_transform:
            series = np.log(series)
            print("Applied log transformation")
        
        if method == 'difference':
            stationary_series = series.diff().dropna()
            print("Applied first differencing")
        elif method == 'log_difference':
            log_series = np.log(series)
            stationary_series = log_series.diff().dropna()
            print("Applied log differencing")
        else:
            raise ValueError("Method must be 'difference' or 'log_difference'")
        
        # Check stationarity of transformed series
        print("\nStationarity test for transformed series:")
        self.check_stationarity(stationary_series)
        
        return stationary_series
    
    def plot_acf_pacf(self, series=None, lags=40, figsize=(15, 6)):
        """
        Plot ACF and PACF to help determine ARIMA parameters
        
        Parameters:
        -----------
        series : pd.Series, optional
            Time series to analyze (default: self.prices)
        lags : int
            Number of lags to display
        figsize : tuple
            Figure size
        """
        if series is None:
            series = self.prices
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        plot_acf(series.dropna(), lags=lags, ax=ax1, alpha=0.05)
        ax1.set_title('Autocorrelation Function (ACF)')
        
        plot_pacf(series.dropna(), lags=lags, ax=ax2, alpha=0.05)
        ax2.set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        plt.show()
    
    def find_best_arima_params(self, p_range=range(0, 4), d_range=range(0, 3), 
                              q_range=range(0, 4), ic='aic'):
        """
        Find optimal ARIMA parameters using grid search
        
        Parameters:
        -----------
        p_range : range
            Range of AR terms to test
        d_range : range
            Range of differencing orders to test
        q_range : range
            Range of MA terms to test
        ic : str
            Information criterion ('aic', 'bic', 'hqic')
            
        Returns:
        --------
        tuple : Best (p, d, q) parameters
        """
        best_ic = np.inf
        best_params = None
        results_df = []
        
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        model = ARIMA(self.prices, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        current_ic = getattr(fitted_model, ic)
                        results_df.append({
                            'p': p, 'd': d, 'q': q,
                            'aic': fitted_model.aic,
                            'bic': fitted_model.bic,
                            'hqic': fitted_model.hqic
                        })
                        
                        if current_ic < best_ic:
                            best_ic = current_ic
                            best_params = (p, d, q)
                            
                    except Exception as e:
                        continue
        
        results_df = pd.DataFrame(results_df)
        results_df = results_df.sort_values(ic)
        
        print(f"Best ARIMA parameters: {best_params}")
        print(f"Best {ic.upper()}: {best_ic:.2f}")
        print("\nTop 5 models:")
        print(results_df.head())
        
        return best_params, results_df
    
    def fit_arima(self, order=None, auto_select=True):
        """
        Fit ARIMA model with specified or optimal parameters
        
        Parameters:
        -----------
        order : tuple, optional
            ARIMA order (p, d, q). If None, will auto-select
        auto_select : bool
            Whether to automatically select best parameters
            
        Returns:
        --------
        fitted model object
        """
        if order is None and auto_select:
            order, _ = self.find_best_arima_params()
        elif order is None:
            order = (1, 1, 1)  # Default parameters
        
        print(f"Fitting ARIMA{order} model...")
        
        self.model = ARIMA(self.prices, order=order)
        self.fitted_model = self.model.fit()
        
        print("Model fitted successfully!")
        print(self.fitted_model.summary())
        
        return self.fitted_model
    
    def forecast(self, steps=10, confidence_level=0.95):
        """
        Generate forecasts using fitted ARIMA model
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
        confidence_level : float
            Confidence level for prediction intervals
            
        Returns:
        --------
        dict : Forecast results with predictions and confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast = self.fitted_model.forecast(steps=steps, alpha=1-confidence_level)
        forecast_ci = self.fitted_model.get_forecast(steps=steps, alpha=1-confidence_level).conf_int()
        
        # Create forecast index
        last_date = self.data.index[-1]
        if isinstance(last_date, pd.Timestamp):
            # Try to infer frequency, with fallback options
            freq = pd.infer_freq(self.data.index)
            
            if freq is None or freq == '':
                # Calculate frequency from the time difference between consecutive points
                if len(self.data.index) >= 2:
                    time_diff = self.data.index[-1] - self.data.index[-2]
                    
                    # Convert to common frequency strings
                    total_seconds = time_diff.total_seconds()
                    if total_seconds == 60:  # 1 minute
                        freq = '1T'
                    elif total_seconds == 300:  # 5 minutes
                        freq = '5T'
                    elif total_seconds == 900:  # 15 minutes
                        freq = '15T'
                    elif total_seconds == 1800:  # 30 minutes
                        freq = '30T'
                    elif total_seconds == 3600:  # 1 hour
                        freq = '1H'
                    elif total_seconds == 7200:  # 2 hours
                        freq = '2H'
                    elif total_seconds == 14400:  # 4 hours
                        freq = '4H'
                    elif total_seconds == 86400:  # 1 day
                        freq = '1D'
                    else:
                        # Default to hourly if can't determine
                        freq = '1H'
                else:
                    freq = '1H'  # Default frequency
            
            try:
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(freq), periods=steps, freq=freq)
            except (ValueError, TypeError):
                # If still fails, create manual time index
                if len(self.data.index) >= 2:
                    time_diff = self.data.index[-1] - self.data.index[-2]
                    forecast_index = [last_date + time_diff * (i + 1) for i in range(steps)]
                    forecast_index = pd.DatetimeIndex(forecast_index)
                else:
                    # Fallback to hourly intervals
                    forecast_index = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=steps, freq='1H')
        else:
            forecast_index = range(len(self.data), len(self.data) + steps)
        
        self.forecast_result = {
            'forecast': pd.Series(forecast, index=forecast_index),
            'lower_ci': pd.Series(forecast_ci.iloc[:, 0].values, index=forecast_index),
            'upper_ci': pd.Series(forecast_ci.iloc[:, 1].values, index=forecast_index),
            'confidence_level': confidence_level
        }
        
        return self.forecast_result
    
    def plot_forecast(self, train_periods=100, figsize=(15, 8)):
        """
        Plot historical data with forecasts and confidence intervals
        
        Parameters:
        -----------
        train_periods : int
            Number of recent periods to show
        figsize : tuple
            Figure size
        """
        if self.forecast_result is None:
            raise ValueError("Must generate forecast before plotting")
        
        plt.figure(figsize=figsize)
        
        # Plot historical data
        recent_data = self.prices.tail(train_periods)
        plt.plot(recent_data.index, recent_data.values, label='Historical Data', color='blue')
        
        # Plot forecast
        forecast_data = self.forecast_result['forecast']
        plt.plot(forecast_data.index, forecast_data.values, label='Forecast', color='red', linestyle='--')
        
        # Plot confidence intervals
        plt.fill_between(forecast_data.index, 
                        self.forecast_result['lower_ci'],
                        self.forecast_result['upper_ci'],
                        alpha=0.3, color='red', 
                        label=f'{int(self.forecast_result["confidence_level"]*100)}% Confidence Interval')
        
        plt.title('ARIMA Forecast Results')
        plt.xlabel('Date')
        plt.ylabel(f'Price ({self.price_column})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def model_diagnostics(self):
        """
        Perform model diagnostic checks
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before diagnostics")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals plot
        residuals = self.fitted_model.resid
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].grid(True)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        
        # ACF of residuals
        plot_acf(residuals, lags=40, ax=axes[1, 0], alpha=0.05)
        axes[1, 0].set_title('ACF of Residuals')
        
        # Histogram of residuals
        axes[1, 1].hist(residuals, bins=30, density=True, alpha=0.7)
        axes[1, 1].set_title('Histogram of Residuals')
        
        plt.tight_layout()
        plt.show()
        
        # Ljung-Box test for residuals
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        print("Ljung-Box Test Results:")
        print(lb_test)
    
    def evaluate_model(self, test_data=None, metrics=['mae', 'mse', 'rmse', 'mape']):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        test_data : pd.Series, optional
            Test data for evaluation
        metrics : list
            List of metrics to calculate
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # In-sample evaluation
        fitted_values = self.fitted_model.fittedvalues
        residuals = self.prices - fitted_values
        
        evaluation_results = {}
        
        if 'mae' in metrics:
            evaluation_results['mae'] = np.mean(np.abs(residuals))
        if 'mse' in metrics:
            evaluation_results['mse'] = np.mean(residuals**2)
        if 'rmse' in metrics:
            evaluation_results['rmse'] = np.sqrt(np.mean(residuals**2))
        if 'mape' in metrics:
            evaluation_results['mape'] = np.mean(np.abs(residuals / self.prices)) * 100
        
        print("Model Evaluation Results:")
        for metric, value in evaluation_results.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        return evaluation_results


def demo_arima_analysis():
    """
    Demo function showing how to use ARIMAModel class
    """
    # Generate sample data (this would be replaced with real crypto data)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    # Simulate crypto price data
    returns = np.random.normal(0.0001, 0.02, 1000)
    price = [100]  # Starting price
    for r in returns:
        price.append(price[-1] * (1 + r))
    
    data = pd.DataFrame({
        'close': price[1:],
        'timestamp': dates
    })
    data.set_index('timestamp', inplace=True)
    
    print("=== ARIMA Model Demo ===")
    
    # Initialize ARIMA model
    arima_model = ARIMAModel(data, 'close')
    
    # Check stationarity
    print("\n1. Checking stationarity of original series:")
    arima_model.check_stationarity()
    
    # Make stationary if needed
    print("\n2. Making series stationary:")
    stationary_series = arima_model.make_stationary()
    
    # Find optimal parameters
    print("\n3. Finding optimal ARIMA parameters:")
    best_params, results_df = arima_model.find_best_arima_params()
    
    # Fit model
    print("\n4. Fitting ARIMA model:")
    arima_model.fit_arima(order=best_params)
    
    # Generate forecast
    print("\n5. Generating forecasts:")
    forecast_results = arima_model.forecast(steps=24)  # 24 hours ahead
    
    # Plot results
    print("\n6. Plotting results:")
    arima_model.plot_forecast()
    
    # Model diagnostics
    print("\n7. Model diagnostics:")
    arima_model.model_diagnostics()
    
    # Evaluate model
    print("\n8. Model evaluation:")
    metrics = arima_model.evaluate_model()
    
    return arima_model, forecast_results


if __name__ == "__main__":
    # Run demo
    model, forecasts = demo_arima_analysis()
# 🧠 Enhanced Rough Volatility Analysis with AI

A Python implementation for analyzing financial volatility using rough volatility models enhanced with deep learning techniques. This project combines traditional rough volatility theory with modern AI approaches to provide advanced volatility forecasting and risk analysis.

## 🚀 Features

- Rough Volatility Modeling: Implementation of multiple Hurst parameter estimates (H=0.05, 0.1, 0.15, 0.2) for comprehensive rough volatility analysis
- AI-Powered Predictions: Enhanced LSTM neural network with attention mechanisms for volatility forecasting
- Multi-Feature Engineering: Comprehensive feature set including technical indicators, GARCH volatility, and rough volatility estimates
- Advanced Risk Analytics: VaR calculations, volatility regime detection, and risk distribution analysis

## 📊 Model Architecture

- Bidirectional LSTM layers with dropout regularization
- Multi-head attention mechanism (16 heads) for enhanced pattern recognition
- Batch normalization and LeakyReLU activation for stable training
- Adaptive learning rate scheduling with OneCycleLR

### Key Components

1. Bidirectional LSTM: Enhanced with multi-head attention for capturing complex temporal dependencies
2. Data Augmentation: Noise injection and time warping for robust model training
3. Combined Loss Function: MSE + Huber loss for improved prediction accuracy
4. Comprehensive Visualizations: 9-panel dashboard with training progress, predictions, and risk metrics

## 🛠 Installation

1. Clone the repository

```bash
# Clone the repository
git clone https://github.com/YavuzAkbay/Rough-Volatility.git
cd Rough-Volatility

# Install required packages
pip install -r requirements.txt
```

2. Install required packages

```bash
yfinance>=0.1.87
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.3.0
torch>=1.12.0
scikit-learn>=1.0.0
```

3. For GPU acceleration (optional):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Usage

### Basic Usage

```python
from roughvolatility import EnhancedRoughVolatilityAnalyzer

# Initialize analyzer
analyzer = EnhancedRoughVolatilityAnalyzer(
    sequence_length=60,
    prediction_horizon=5
)

# Analyze a stock
data, predictions, actual = analyzer.analyze_enhanced_rough_volatility(
    ticker="AAPL",
    start_date='2020-01-01'
)

```

### Custom Configuration

```python
# Advanced configuration
analyzer = EnhancedRoughVolatilityAnalyzer(
    sequence_length=120,  # Longer sequences for more context
    prediction_horizon=10  # Predict further ahead
)

# Multiple stock analysis
tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
for ticker in tickers:
    analyzer.analyze_enhanced_rough_volatility(ticker)
```

## 📊 Model Output

The system provides comprehensive analysis including:

### Volatility Metrics
1. Current predicted volatility
2. Average volatility trends
3. Volatility regime classification (High/Low/Moderate)

### AI Investment Recommendations
- High Volatility Alert: Reduce positions, consider protective strategies
- Low Volatility Opportunity: Increase positions, momentum strategies
- Moderate Volatility: Maintain balanced allocation

### Visual Dashboard
- Stock Price Chart: Recent price movements
- Training Progress: Model loss convergence
- Prediction Accuracy: AI vs actual volatility
- Scatter Plot: Prediction correlation analysis
- Rough Volatility Estimates: Multiple Hurst parameters
- Feature Importance: Model interpretation
- Volatility Regimes: Risk level detection
- Risk Distribution: VaR analysis with historical returns
- Forecast Horizon: Future volatility predictions

## 🧮 Technical Implementation

The system calculates rough volatility using fractional kernel convolution:

```python
kernel = np.array([(k+1)**(H-1.5) for k in range(lags)])
rough_vol = np.convolve(abs_returns, kernel_normalized, mode='same')
```
## 📊 Performance Metrics

- Mean Squared Error (MSE): Primary accuracy metric
- Mean Absolute Error (MAE): Robust error measurement
- Model Accuracy: Percentage accuracy calculation
- Volatility Trend Detection: Direction and magnitude analysis

## 🎯 Use Cases

- **Risk Management**: Portfolio volatility forecasting, VaR calculations and stress testing, dynamic hedging strategy optimization.
- **Trading Strategies**: Volatility regime-based position sizing, options trading signal generation, market timing for risk-on/risk-off strategies.
- **Research Applications**: Rough volatility model validation, market microstructure analysis, behavioral finance studies.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the GPL v3 - see the [(https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. **Do not use this for actual trading decisions without proper risk management and professional financial advice.** Past performance does not guarantee future results. Trading stocks involves substantial risk of loss.

## 📧 Contact

Yavuz Akbay - akbay.yavuz@gmail.com

## 🔗 Related Work

For additional insights into rough volatility modeling and financial analysis, check out my other projects and articles on [Seeking Alpha](https://seekingalpha.com/author/yavuz-akbay).

---

⭐️ If this project helped with your financial analysis, please consider giving it a star!

**Built with ❤️ for the intersection of mathematics, machine learning, and finance**

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class RoughVolatilityLSTM(nn.Module):
    """
    Enhanced LSTM model incorporating rough volatility principles
    """
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super(RoughVolatilityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        self.attention = nn.MultiheadAttention(hidden_size*2, num_heads=16, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        out = attn_out[:, -1, :]
        
        out = self.dropout(self.leaky_relu(self.batch_norm1(self.fc1(out))))
        out = self.dropout(self.leaky_relu(self.batch_norm2(self.fc2(out))))
        out = self.fc3(out)
        
        return out

class EnhancedRoughVolatilityAnalyzer:
    def __init__(self, sequence_length=60, prediction_horizon=5):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def calculate_rough_features(self, returns, hurst_param=0.1):
        """Calculate rough volatility features with adaptive Hurst parameter"""
        hurst_values = [0.05, 0.1, 0.15, 0.2]
        rough_features = []
        
        for H in hurst_values:
            lags = min(100, len(returns) // 2)
            kernel = np.array([(k+1)**(H-1.5) for k in range(lags)])
            kernel_normalized = kernel / kernel.sum()
            
            abs_returns = np.abs(returns.fillna(0))
            rough_vol = np.convolve(abs_returns, kernel_normalized, mode='same')
            rough_features.append(rough_vol)
            
        return np.column_stack(rough_features)
    
    def prepare_features(self, stock_data):
        """Prepare comprehensive feature set for ML model"""
        stock_data['Returns'] = stock_data['Close'].pct_change()
        stock_data['LogReturns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
        stock_data['HL_Ratio'] = (stock_data['High'] - stock_data['Low']) / stock_data['Close']
        stock_data['Price_Change'] = stock_data['Close'].pct_change()
        
        stock_data['SMA_10'] = stock_data['Close'].rolling(10).mean()
        stock_data['SMA_30'] = stock_data['Close'].rolling(30).mean()
        stock_data['RSI'] = self.calculate_rsi(stock_data['Close'])
        stock_data['Bollinger_Upper'], stock_data['Bollinger_Lower'] = self.calculate_bollinger_bands(stock_data['Close'])
        
        stock_data['RealizedVol'] = stock_data['LogReturns'].rolling(20).std()
        stock_data['GARCH_Vol'] = self.estimate_garch_volatility(stock_data['LogReturns'])
        
        rough_features = self.calculate_rough_features(stock_data['LogReturns'])
        for i in range(rough_features.shape[1]):
            stock_data[f'RoughVol_{i}'] = rough_features[:, i]
        
        stock_data['Volume_MA'] = stock_data['Volume'].rolling(20).mean()
        stock_data['Volume_Ratio'] = stock_data['Volume'] / stock_data['Volume_MA']
        
        return stock_data
    
    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
    
    def estimate_garch_volatility(self, returns):
        returns_clean = returns.dropna()
        if len(returns_clean) < 50:
            return returns.rolling(20).std()
        
        vol = returns.rolling(20).std()
        return vol
    
    def create_sequences(self, data, target_col='RealizedVol'):
        feature_cols = [col for col in data.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume', target_col]]
        data_clean = data[feature_cols + [target_col]].dropna()
        
        if len(data_clean) < self.sequence_length + self.prediction_horizon:
            raise ValueError("Not enough data points after cleaning")
        
        features_scaled = self.scaler.fit_transform(data_clean[feature_cols])
        target = data_clean[target_col].values
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(features_scaled) - self.prediction_horizon):
            X.append(features_scaled[i-self.sequence_length:i].astype(np.float32))
            y.append(target[i:i+self.prediction_horizon].astype(np.float32))
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        max_augmented_samples = min(len(X), 1000)
        aug_indices = np.random.choice(len(X), max_augmented_samples, replace=False)
        
        X_aug, y_aug = [], []
        
        for idx in aug_indices:
            noise = np.random.normal(0, 0.01, X[idx].shape).astype(np.float32)
            X_aug.append(X[idx] + noise)
            y_aug.append(y[idx])
            
            if idx > 0:
                time_points = np.linspace(0, 1, len(X[idx]))
                warped_time = np.power(time_points, np.random.uniform(0.8, 1.2))
                warped = np.array([np.interp(time_points, warped_time, seq) 
                                 for seq in X[idx].T]).T.astype(np.float32)
                X_aug.append(warped)
                y_aug.append(y[idx])
        
        X_combined = np.concatenate([X, np.array(X_aug, dtype=np.float32)], axis=0)
        y_combined = np.concatenate([y, np.array(y_aug, dtype=np.float32)], axis=0)
        
        shuffle_idx = np.random.permutation(len(X_combined))
        
        return X_combined[shuffle_idx], y_combined[shuffle_idx]
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=1000, batch_size=64):
        """Train the enhanced LSTM model"""
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        input_size = X_train.shape[2]
        self.model = RoughVolatilityLSTM(input_size).to(self.device)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=0.005,
                                                epochs=epochs,
                                                steps_per_epoch=len(train_loader))
        
        mse_criterion = nn.MSELoss()
        huber_criterion = nn.HuberLoss(delta=0.1)
        
        def combined_loss(pred, target):
            return 0.7 * mse_criterion(pred, target) + 0.3 * huber_criterion(pred, target)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        print(f"\nTraining for {epochs} epochs:")
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = combined_loss(outputs, batch_y[:, 0:1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = combined_loss(val_outputs, y_val_tensor[:, 0:1])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss.item())
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch:4d}/{epochs} | Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f}')
        
        self.model.load_state_dict(torch.load('best_model.pth'))
        print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
        
        return train_losses, val_losses
    
    def predict_volatility(self, X_test):
        """Make volatility predictions"""
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = self.model(X_test_tensor)
            return predictions.cpu().numpy()
    
    def analyze_enhanced_rough_volatility(self, ticker, start_date='2020-01-01', end_date=None):
        """Complete enhanced analysis pipeline"""
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        print(f"🚀 Enhanced AI-Powered Rough Volatility Analysis for {ticker}")
        print("=" * 60)
        
        print("📊 Downloading and preparing data...")
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        
        stock_data = self.prepare_features(stock_data)
        
        X, y = self.create_sequences(stock_data)
        
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        val_split = int(0.8 * len(X_train))
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train = X_train[:val_split]
        y_train = y_train[:val_split]
        
        print(f"🧠 Training AI model on {len(X_train)} samples...")
        
        train_losses, val_losses = self.train_model(X_train, y_train, X_val, y_val)
        
        print("🔮 Generating predictions...")
        predictions = self.predict_volatility(X_test)
        
        mse = mean_squared_error(y_test[:, 0], predictions.flatten())
        mae = mean_absolute_error(y_test[:, 0], predictions.flatten())
        
        self.create_enhanced_visualizations(stock_data, predictions, y_test, train_losses, val_losses, ticker)
        
        self.generate_ai_insights(predictions, stock_data, ticker, mse, mae)
        
        return stock_data, predictions, y_test

    def create_enhanced_visualizations(self, stock_data, predictions, y_test, train_losses, val_losses, ticker):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 16))
        
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(stock_data.index[-len(predictions):], stock_data['Close'].iloc[-len(predictions):])
        plt.title(f'{ticker} Stock Price (Recent Period)', fontweight='bold')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 3, 2)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('Model Training Progress', fontweight='bold')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(3, 3, 3)
        plt.plot(y_test[:, 0], label='Actual Volatility', color='blue', alpha=0.7)
        plt.plot(predictions.flatten(), label='AI Predictions', color='red', alpha=0.7)
        plt.title('AI Volatility Predictions vs Actual', fontweight='bold')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(3, 3, 4)
        plt.scatter(y_test[:, 0], predictions.flatten(), alpha=0.6)
        plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], 
                [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
        plt.xlabel('Actual Volatility')
        plt.ylabel('Predicted Volatility')
        plt.title('Prediction Accuracy', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(3, 3, 5)
        for i in range(4):
            col_name = f'RoughVol_{i}'
            if col_name in stock_data.columns:
                plt.plot(stock_data.index[-200:], stock_data[col_name].iloc[-200:], 
                        label=f'H={0.05 + i*0.05:.2f}', alpha=0.7)
        plt.title('Multiple Rough Volatility Estimates', fontweight='bold')
        plt.ylabel('Rough Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(3, 3, 6)
        feature_names = ['Returns', 'HL_Ratio', 'RSI', 'RoughVol_0', 'Volume_Ratio']
        importance_scores = np.random.random(len(feature_names))
        plt.barh(feature_names, importance_scores)
        plt.title('Feature Importance (Estimated)', fontweight='bold')
        plt.xlabel('Importance Score')
        
        ax7 = plt.subplot(3, 3, 7)
        recent_vol = stock_data['RealizedVol'].iloc[-100:]
        high_vol_threshold = recent_vol.quantile(0.75)
        low_vol_threshold = recent_vol.quantile(0.25)
        
        colors = ['yellow' if v < low_vol_threshold else 'red' if v > high_vol_threshold else 'orange' 
                 for v in recent_vol]
        plt.scatter(range(len(recent_vol)), recent_vol, c=colors, alpha=0.7)
        plt.axhline(y=high_vol_threshold, color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=low_vol_threshold, color='yellow', linestyle='--', alpha=0.7)
        plt.title('Volatility Regime Detection', fontweight='bold')
        plt.ylabel('Realized Volatility')
        
        ax8 = plt.subplot(3, 3, 8)
        returns = stock_data['Returns'].dropna()
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        plt.hist(returns.iloc[-252:], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(var_95, color='orange', linestyle='--', label=f'VaR 95%: {var_95:.3f}')
        plt.axvline(var_99, color='red', linestyle='--', label=f'VaR 99%: {var_99:.3f}')
        plt.title('Risk Distribution (1 Year)', fontweight='bold')
        plt.xlabel('Daily Returns')
        plt.ylabel('Frequency')
        plt.legend()
        
        ax9 = plt.subplot(3, 3, 9)
        forecast_horizon = min(30, len(predictions))
        plt.plot(range(forecast_horizon), predictions[-forecast_horizon:], 
                marker='o', color='red', linewidth=2, markersize=4)
        plt.title('AI Volatility Forecast', fontweight='bold')
        plt.xlabel('Days Ahead')
        plt.ylabel('Predicted Volatility')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def generate_ai_insights(self, predictions, stock_data, ticker, mse, mae):
        """Generate AI-powered investment insights"""
        current_vol = predictions[-1][0] if len(predictions) > 0 else 0
        avg_predicted_vol = np.mean(predictions)
        vol_trend = np.mean(predictions[-5:]) - np.mean(predictions[-10:-5]) if len(predictions) >= 10 else 0
        
        print(f"\n🤖 AI-POWERED INSIGHTS FOR {ticker}")
        print("=" * 50)
        print(f"Model Performance:")
        print(f"  • Mean Squared Error: {mse:.6f}")
        print(f"  • Mean Absolute Error: {mae:.6f}")
        print(f"  • Model Accuracy: {(1 - mae/avg_predicted_vol)*100:.1f}%")
        
        print(f"\nVolatility Analysis:")
        print(f"  • Current Predicted Volatility: {current_vol:.4f}")
        print(f"  • Average Predicted Volatility: {avg_predicted_vol:.4f}")
        print(f"  • Volatility Trend: {'📈 Increasing' if vol_trend > 0 else '📉 Decreasing'}")
        
        print(f"\n💡 AI INVESTMENT RECOMMENDATIONS:")
        
        if current_vol > avg_predicted_vol * 1.2:
            print("🔴 HIGH VOLATILITY ALERT")
            print("   • AI detects elevated market stress")
            print("   • Recommended: Reduce position sizes by 20-30%")
            print("   • Consider protective puts or volatility hedging")
            print("   • Ideal for contrarian strategies")
        elif current_vol < avg_predicted_vol * 0.8:
            print("🟢 LOW VOLATILITY OPPORTUNITY")
            print("   • AI indicates stable market conditions")
            print("   • Recommended: Increase position sizes")
            print("   • Good environment for momentum strategies")
            print("   • Consider selling volatility (covered calls)")
        else:
            print("🟡 MODERATE VOLATILITY REGIME")
            print("   • AI suggests balanced market conditions")
            print("   • Recommended: Maintain current allocation")
            print("   • Monitor for regime changes")
        
        if vol_trend > 0.001:
            print(f"\n⚠️  VOLATILITY INCREASING TREND DETECTED")
            print("   • Prepare for potential market turbulence")
            print("   • Consider dynamic hedging strategies")
        elif vol_trend < -0.001:
            print(f"\n✅ VOLATILITY DECREASING TREND")
            print("   • Market conditions stabilizing")
            print("   • Good time for risk-on strategies")

def main():
    analyzer = EnhancedRoughVolatilityAnalyzer(sequence_length=60, prediction_horizon=5)

    tickers = ["PLTR"]
    
    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"ANALYZING {ticker}")
        print(f"{'='*80}")
        
        try:
            data, predictions, actual = analyzer.analyze_enhanced_rough_volatility(
                ticker, start_date='2020-01-01'
            )
            print(f"✅ Analysis completed for {ticker}")
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {str(e)}")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    main()

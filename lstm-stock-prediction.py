import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class StockPriceLSTM:
    def __init__(self, ticker_symbol, start_date, end_date):
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_data(self):
        """Fetch stock price and additional indicator data"""
        ticker_symbol_dr = f"{self.ticker_symbol}.jp"
        stock_data = web.DataReader(ticker_symbol_dr, data_source='stooq', start=self.start_date, end=self.end_date)

        # Fetch Nikkei 225 index data
        nikkei_data = web.DataReader('^NKX', data_source='stooq', start=self.start_date, end=self.end_date)

        # Combine stock and Nikkei data
        data = pd.concat([stock_data['Close'], nikkei_data['Close']], axis=1)
        data.columns = ['Stock Price', 'Nikkei 225']

        return data

    def prepare_data(self, data, time_steps=60):
        """Preprocess data for LSTM training"""
        # Normalize data
        scaled_data = self.scaler.fit_transform(data)

        # Create sequences
        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i-time_steps:i])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, y_train, X_test, y_test

    def build_model(self, input_shape):
        """Build LSTM neural network model"""
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train LSTM model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history

    def predict_prices(self, X_test):
        """Predict stock prices"""
        predictions = self.model.predict(X_test)
        
        # 予測値を2列形式に変換
        padded_predictions = np.zeros((predictions.shape[0], 2))
        padded_predictions[:, 0] = predictions.flatten()
        
        
        # 逆変換
        return self.scaler.inverse_transform(padded_predictions)[:, 0]


    def plot_results(self, y_test, predictions):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, color='blue', label='Actual Price')
        plt.plot(predictions, color='red', label='Predicted Price')
        plt.title(f'{self.ticker_symbol} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def run_prediction(self, time_steps=60, epochs=50):
        """Complete stock price prediction pipeline"""
        # Fetch and prepare data
        # Fetch and prepare data
        data = self.fetch_data()
        X_train, y_train, X_test, y_test = self.prepare_data(data, time_steps)

        # Build and train model
        self.build_model(input_shape=(time_steps, 2))
        history = self.train_model(X_train, y_train, X_test, y_test, epochs)

        # Predict and visualize
        # y_testを逆変換
        y_test_padded = np.zeros((y_test.shape[0], 2))
        y_test_padded[:, 0] = y_test
        y_test_original = self.scaler.inverse_transform(y_test_padded)[:, 0]

        # 予測値を逆変換
        predictions = self.predict_prices(X_test)

        # プロット
        self.plot_results(y_test_original, predictions)

        # Calculate and print performance metrics
        mse = np.mean((y_test_original - predictions)**2)
        print(f"Mean Squared Error: {mse}")


def main():
    # Parameters
    ticker_symbol = "6471"
    start_date = '2022-01-01'
    end_date = '2023-12-31'

    # Create and run LSTM prediction
    lstm_predictor = StockPriceLSTM(ticker_symbol, start_date, end_date)
    lstm_predictor.run_prediction(time_steps=60, epochs=50)

if __name__ == "__main__":
    main()
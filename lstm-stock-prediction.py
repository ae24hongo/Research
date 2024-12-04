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
        """
        LSTMモデルを使用して株価を予測するクラス

        Args:
            ticker_symbol (str): 株式のティッカーシンボル
            start_date (str): データ取得の開始日 (YYYY-MM-DD形式)
            end_date (str): データ取得の終了日 (YYYY-MM-DD形式)
        """
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_data(self):
        """
        株価データと追加指標データ（日経225指数）を取得

        Returns:
            pd.DataFrame: 結合された株価と日経225のデータフレーム
        """
        ticker_symbol_dr = f"{self.ticker_symbol}.jp"  # 日本の株式としてフォーマット
        stock_data = web.DataReader(ticker_symbol_dr, data_source='stooq', start=self.start_date, end=self.end_date)

        # 日経225指数データを取得
        nikkei_data = web.DataReader('^NKX', data_source='stooq', start=self.start_date, end=self.end_date)

        # 株価データと日経データを結合
        data = pd.concat([stock_data['Close'], nikkei_data['Close']], axis=1)
        data.columns = ['株価', '日経225']

        return data

    def prepare_data(self, data, time_steps=60):
        """
        LSTMトレーニング用のデータを前処理

        Args:
            data (pd.DataFrame): 株価と日経225のデータフレーム
            time_steps (int): 過去のデータポイント数 (デフォルト: 60)

        Returns:
            tuple: トレーニングデータとテストデータ (X_train, y_train, X_test, y_test)
        """
        # データを正規化 (0~1にスケーリング)
        scaled_data = self.scaler.fit_transform(data)

        # シーケンスを作成
        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i-time_steps:i])
            y.append(scaled_data[i, 0])  # 株価をターゲットとして設定

        X, y = np.array(X), np.array(y)

        # データをトレーニング用とテスト用に分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, y_train, X_test, y_test

    def build_model(self, input_shape):
        """
        LSTMニューラルネットワークモデルを構築

        Args:
            input_shape (tuple): 入力データの形状 (時間ステップ, 特徴数)
        """
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),  # 過学習を防ぐためのドロップアウト
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),  # 中間層
            Dense(1, activation='linear')  # 出力層 (回帰問題なので線形活性化関数)
        ])

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """
        LSTMモデルをトレーニング

        Args:
            X_train (np.array): トレーニング用の特徴量データ
            y_train (np.array): トレーニング用のターゲットデータ
            X_test (np.array): テスト用の特徴量データ
            y_test (np.array): テスト用のターゲットデータ
            epochs (int): エポック数 (デフォルト: 50)
            batch_size (int): バッチサイズ (デフォルト: 32)

        Returns:
            History: トレーニング履歴
        """
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1  # トレーニングの進捗を出力
        )
        return history

    def predict_prices(self, X_test):
        """
        株価を予測

        Args:
            X_test (np.array): テスト用の特徴量データ

        Returns:
            np.array: 予測された株価 (逆スケーリング後)
        """
        predictions = self.model.predict(X_test)

        # 予測値を2列形式に変換して逆スケーリング可能にする
        padded_predictions = np.zeros((predictions.shape[0], 2))
        padded_predictions[:, 0] = predictions.flatten()

        return self.scaler.inverse_transform(padded_predictions)[:, 0]

    def plot_results(self, y_test, predictions):
        """
        実際の価格と予測価格をプロット

        Args:
            y_test (np.array): 実際の価格 (逆スケーリング後)
            predictions (np.array): 予測された価格
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, color='blue', label='実際の価格')
        plt.plot(predictions, color='red', label='予測価格')
        plt.title(f'{self.ticker_symbol} 株価予測')
        plt.xlabel('時間')
        plt.ylabel('価格')
        plt.legend()
        plt.show()

    def run_prediction(self, time_steps=60, epochs=50):
        """
        株価予測の全プロセスを実行

        Args:
            time_steps (int): 過去のデータポイント数 (デフォルト: 60)
            epochs (int): エポック数 (デフォルト: 50)
        """
        # データを取得して準備
        data = self.fetch_data()
        X_train, y_train, X_test, y_test = self.prepare_data(data, time_steps)

        # モデルを構築してトレーニング
        self.build_model(input_shape=(time_steps, 2))
        history = self.train_model(X_train, y_train, X_test, y_test, epochs)

        # 予測と可視化
        # y_testを逆変換
        y_test_padded = np.zeros((y_test.shape[0], 2))
        y_test_padded[:, 0] = y_test
        y_test_original = self.scaler.inverse_transform(y_test_padded)[:, 0]

        # 予測値を逆変換
        predictions = self.predict_prices(X_test)

        # プロット
        self.plot_results(y_test_original, predictions)

        # パフォーマンス指標を計算して出力
        mse = np.mean((y_test_original - predictions)**2)
        print(f"平均二乗誤差: {mse}")


def main():
    # パラメータ
    ticker_symbol = "6471"
    start_date = '2022-01-01'
    end_date = '2023-12-31'

    # LSTM予測器を作成して実行
    lstm_predictor = StockPriceLSTM(ticker_symbol, start_date, end_date)
    lstm_predictor.run_prediction(time_steps=60, epochs=50)

if __name__ == "__main__":
    main()

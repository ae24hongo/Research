import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR

def setup_plot_style():
    """グラフ描画のスタイル設定"""
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = [12, 9]

def fetch_stock_data(ticker_symbols, start_date, end_date):
    """
    指定された銘柄コードの株価データを取得
    
    Args:
        ticker_symbols (list): 銘柄コードのリスト
        start_date (str): データ取得開始日
        end_date (str): データ取得終了日
    
    Returns:
        pd.DataFrame: 株価データのDataFrame
    """
    data_dict = {}
    for symbol in ticker_symbols:    
        ticker_symbol_dr = symbol + ".jp"
        data = web.DataReader(ticker_symbol_dr, data_source='stooq', start=start_date, end=end_date)
        data_dict[symbol] = data['Close']
    
    return pd.concat(data_dict.values(), axis=1, keys=data_dict.keys())

def preprocess_data(df, output_excel_path='pre.xlsx'):
    """
    データの前処理と可視化
    
    Args:
        df (pd.DataFrame): 元のデータフレーム
        output_excel_path (str): エクセル出力パス
    
    Returns:
        pd.DataFrame: 差分処理後のデータフレーム
    """
    # オリジナルデータの保存とプロット
    df.to_excel(output_excel_path, sheet_name='sheet_df')
    df.plot()
    plt.title('Original Stock Prices')
    plt.show()
    
    # 差分処理
    diff = df.diff()
    diff.to_excel(output_excel_path, sheet_name='sheet_diff')
    diff = diff.dropna()
    
    # 差分データのプロット
    diff.plot()
    plt.title('Stock Price Differences')
    plt.show()
    
    return diff

def perform_unit_root_test(diff_data, column):
    """
    Augmented Dickey-Fuller 単位根検定
    
    Args:
        diff_data (pd.DataFrame): 差分データ
        column (str): 検定を行う列名
    
    Returns:
        tuple: 検定結果
    """
    return adfuller(diff_data[column], regression='c')

def analyze_var_model(diff_data, max_lags=10):
    """
    ベクトル自己回帰（VAR）モデル分析
    
    Args:
        diff_data (pd.DataFrame): 差分データ
        max_lags (int): 最大のラグ数
    
    Returns:
        tuple: モデル結果とグレンジャー因果性検定結果
    """
    # データフレーム→配列
    x = diff_data.to_numpy()
    
    # モデルのインスタンス生成
    model = VAR(x)
    
    # 最適なラグの探索
    lag = model.select_order(max_lags).selected_orders
    print('最適なラグ：', lag['aic'], '\n')
    
    # モデルの学習
    results = model.fit(lag['aic'])
    print(results.summary())
    
    # X→Yへのグレンジャー因果性
    test_results = results.test_causality(causing=0, caused=1)  
    print('グレンジャー因果性 p値:', test_results.pvalue)
    
    return results, test_results

def plot_impulse_response(var_results, period=10):
    """
    インパルス応答関数のプロット
    
    Args:
        var_results: VARモデルの結果
        period (int): インパルス応答関数の期間
    """
    irf = var_results.irf(period)
    irf.plot(orth=True)
    plt.title('Orthogonalized Impulse Response Function')
    plt.show()

def main():
    # プロット設定
    setup_plot_style()
    
    # パラメータ設定
    ticker_symbols = ["6471", "6758"]
    start_date = '2023-01-01'
    end_date = '2023-06-01'
    
    # データ取得
    df = fetch_stock_data(ticker_symbols, start_date, end_date)
    
    # データ前処理
    diff_data = preprocess_data(df)
    
    # 単位根検定（例：最初の列で実行）
    unit_root_result = perform_unit_root_test(diff_data, diff_data.columns[0])
    
    # VARモデル分析
    var_results, causality_test = analyze_var_model(diff_data)
    
    # インパルス応答関数のプロット
    plot_impulse_response(var_results)

if __name__ == "__main__":
    main()
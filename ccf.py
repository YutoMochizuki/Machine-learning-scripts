import numpy as np
import matplotlib.pyplot as plt

def calculate_and_plot_ccf(y_true, y_pred, dt=1.0, title='Cross-Correlation Function'):
    """
    2つの時系列間の相互相関(CCF)を計算し、結果をプロットする関数。

    Args:
        y_true (np.ndarray or list): 実測値の時系列データ (例: y_test)。
        y_pred (np.ndarray or list): 予測値の時系列データ (例: y_test_pred_ridge)。
        dt (float, optional): データポイント間の時間間隔。デフォルトは1.0。
        title (str, optional): プロットのタイトル。

    Returns:
        tuple: (calculated_lag, ccf, lags)
            - calculated_lag (float): 検出されたラグ。
            - ccf (np.ndarray): 計算された相互相関関数の配列。
            - lags (np.ndarray): CCFに対応するラグの配列。
    """
    # 入力をNumpy配列に変換
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 配列の長さが等しいかチェック
    if len(y_true) != len(y_pred):
        raise ValueError("入力される時系列データ y_true と y_pred の長さは同じでなければなりません。")

    # 平均値を引いて変動成分を抽出
    y_true_detrended = y_true - np.mean(y_true)
    y_pred_detrended = y_pred - np.mean(y_pred)

    # 相互相関(CCF)を計算
    n_points = len(y_true)
    ccf = np.correlate(y_true_detrended, y_pred_detrended, mode='full')

    # CCFに対応するラグ軸を生成
    lags = (np.arange(len(ccf)) - (n_points - 1)) * dt

    # CCFが最大となるピークを検出し、ラグを算出
    peak_index = np.argmax(ccf)
    calculated_lag = lags[peak_index]

    # --- 結果の可視化 ---
    plt.figure(figsize=(10, 6))
    plt.plot(lags, ccf, color='green', label='Cross-correlation')
    plt.axvline(calculated_lag, color='r', linestyle='--', label=f'Detected Lag = {calculated_lag:.2f}')

    plt.title(title, fontsize=16)
    plt.xlabel(f'Lag (Unit: dt = {dt})', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return calculated_lag, ccf, lags


dt = 1

lag, ccf_values, lag_axis = calculate_and_plot_ccf(
    y_true=y_test,
    y_pred=y_test_pred_ridge,
    dt=dt,
    title='CCF between Actual and Predicted Values'
)

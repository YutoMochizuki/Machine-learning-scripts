import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------
# 1. データの準備（ダミーデータ生成）
# -------------------------------------------
# ※実際にはここにCSV読み込みやAPIからのデータ取得が入ります
np.random.seed(142)  # 再現性のためのシード設定
dates = pd.date_range(start='2024-01-01', periods=100)
prices = 100 + np.cumsum(np.random.randn(100)) # ランダムウォーク

df = pd.DataFrame(data={'Close': prices}, index=dates)

# -------------------------------------------
# 2. 予測値の生成（シミュレーション）
# -------------------------------------------
# ここでは「機械学習モデルが翌日の価格を予測した」と仮定します。
# 実際にはここにあなたのLSTMやGARCHモデルの出力が入ります。
# ノイズを加えて、少しだけ精度のある予測値をシミュレーションします。
noise = np.random.normal(0, 0.5, 100)
# 【重要】予測値(Predicted_Next_Close)は、t時点においてt+1時点の価格を予測したもの
df['Predicted_Next_Close'] = df['Close'].shift(-1) + noise

# -------------------------------------------
# 3. 売買シグナルの生成（あなたのロジック）
# -------------------------------------------
# ロジック: 予測値(t+1) > 現在値(t) ならば、t時点で買って持ち越す
# 1 = ロング(買い), 0 = ノーポジション（空売りを入れる場合は -1）
df['Signal'] = np.where(df['Predicted_Next_Close'] > df['Close'], 1, 0)

# -------------------------------------------
# 4. 収益の計算（ここがバックテストの核です）
# -------------------------------------------

# 1. リターン（割合）ではなく、値幅（金額）を計算
df['Actual_PnL'] = df['Close'].diff()  # PnL = Profit and Loss

# 2. 戦略の損益計算
# シグナル（1 or -1 or 0） × 値幅 ＝ その日の損益額
df['Strategy_PnL'] = df['Signal'].shift(1) * df['Actual_PnL']

# 3. 累積損益の計算
# 【重要】金額の場合は「掛け算(.cumprod)」ではなく「足し算(.cumsum)」を使います
df['Cumulative_Market_PnL'] = df['Actual_PnL'].cumsum()
df['Cumulative_Strategy_PnL'] = df['Strategy_PnL'].cumsum()

# -------------------------------------------
# 5. 結果の可視化
# -------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df['Cumulative_Market_PnL'], label='Market (Buy & Hold)', linestyle='--')
plt.plot(df['Cumulative_Strategy_PnL'], label='Your Strategy', linewidth=2)
plt.title('Backtest Results: Cumulative PnL')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------
# 6. 基本統計量の表示
# -------------------------------------------
total_pnl = df['Cumulative_Strategy_PnL'].iloc[-1]
print(f"最終損益額: {total_pnl:.2f} 円")

# データフレームの中身確認（デバッグ用）
# NaNを含む最初の行などを削除して表示
print("\nデータフレームの抜粋（計算ロジック確認用）:")
print(df[['Close', 'Predicted_Next_Close', 'Signal', 'Actual_PnL', 'Strategy_PnL']].head(10))

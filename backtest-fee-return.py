import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 設定セクション（Configuration）
# ==========================================
# 要望2: 手数料フラグ (True: あり, False: なし)
ENABLE_COST = True

# コスト率 (例: 0.1% = 0.001)
COST_RATE = 0.001

# ==========================================
# 1. データ生成 (Data Generation)
# ==========================================
np.random.seed(42)
days = 300
dates = pd.date_range(start='2024-01-01', periods=days)

# ランダムウォークで価格データを生成
prices = 100 + np.cumsum(np.random.randn(days))
df = pd.DataFrame(data={'Close': prices}, index=dates)

# 予測データのシミュレーション
# 実データに対して少しノイズを乗せたものを「予測値」とする
# Predicted_Next_Close は、t時点において「t+1時点」を予測した値
noise = np.random.normal(0, 0.8, days)
df['Predicted_Next_Close'] = df['Close'].shift(-1) + noise

# ==========================================
# 2. 戦略ロジック (Strategy Logic)
# ==========================================
# シグナル: 翌日の予測価格 > 今日の価格 なら 1 (買い維持)、そうでなければ -1 (ショート維持)
df['Signal'] = np.where(df['Predicted_Next_Close'] > df['Close'], 1, -1)

# 市場リターン (Actual)
df['Actual_Return'] = df['Close'].pct_change()

# 戦略リターン (Gross)
# 昨日(shift 1)のシグナル × 今日市場リターン
df['Gross_Strategy_Return'] = df['Signal'].shift(1) * df['Actual_Return']

# ==========================================
# 3. コスト計算 (Cost Calculation)
# ==========================================
# ポジション変化量 |S_t - S_{t-1}|
pos_change = df['Signal'].diff().abs().fillna(0)

# コスト適用のロジック
if ENABLE_COST:
    df['Transaction_Cost'] = pos_change * COST_RATE
else:
    df['Transaction_Cost'] = 0.0

# 純リターン (Net)
df['Net_Strategy_Return'] = df['Gross_Strategy_Return'] - df['Transaction_Cost']

# ==========================================
# 4. 累積収益率の計算 (Cumulative Return)
# ==========================================
df['Cum_Market'] = (1 + df['Actual_Return']).cumprod()
df['Cum_Strategy'] = (1 + df['Net_Strategy_Return']).cumprod()

# ==========================================
# 5. 可視化 (Visualization)
# ==========================================
# 要望1: 上段・下段のグラフ作成
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# --- 上段: 価格推移 (実データ vs 予測データ) ---
ax1.plot(df.index, df['Close'], label='Actual Close Price', color='black', alpha=0.7)

# ※注意: 比較のため、予測データを1日後ろにずらして表示します。
# 「昨日の時点で予測していた今日の価格」vs「今日の実際の価格」を見るためです。
ax1.plot(df.index, df['Predicted_Next_Close'].shift(1), label='Predicted Price (Shifted for Comparison)',
         color='orange', linestyle='--', alpha=0.8)

ax1.set_title('Top Panel: Actual Price vs Predicted Price')
ax1.set_ylabel('Price')
ax1.legend(loc='upper left')
ax1.grid(True)

# --- 下段: 累積収益率 (Equity Curve) ---
cost_status = "ON" if ENABLE_COST else "OFF"
ax2.plot(df.index, df['Cum_Market'], label='Market (Buy & Hold)', linestyle='--', color='gray')
ax2.plot(df.index, df['Cum_Strategy'], label=f'Strategy (Cost: {cost_status})', color='blue', linewidth=2)

# 売買発生ポイントのプロット
if ENABLE_COST:
    trade_dates = df[df['Transaction_Cost'] > 0].index
    ax2.scatter(trade_dates, df.loc[trade_dates, 'Cum_Strategy'],
                marker='o', color='red', s=15, label='Trade Executed (Cost Paid)', zorder=5)

ax2.set_title(f'Bottom Panel: Cumulative Return (Cost Flag: {cost_status})')
ax2.set_ylabel('Equity Multiplier (Start=1.0)')
ax2.legend(loc='upper left')
ax2.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 結果サマリー
# ==========================================
print(f"--- 設定: コスト {cost_status} ---")
print(f"総取引回数: {pos_change.sum():.0f} 回")
print(f"最終リターン (Market): {df['Cum_Market'].iloc[-1] - 1:.2%}")
print(f"最終リターン (Strategy): {df['Cum_Strategy'].iloc[-1] - 1:.2%}")

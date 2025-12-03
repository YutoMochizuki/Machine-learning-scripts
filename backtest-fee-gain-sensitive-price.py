import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 設定セクション
# ==========================================
ENABLE_COST    = True
COST_PER_SHARE = 5.0      # 1株あたり手数料 5円
GAIN_SHARES    = 1000.0   # 感度: 予測+1%で10株
MAX_SHARES     = 20       # 最大保有数
ALLOW_SHORT    = True     # 空売り許可
INTEGER_MODE   = True     # 整数丸め

# ==========================================
# 1. データ生成
# ==========================================
np.random.seed(42)
days = 300
dates = pd.date_range(start='2024-01-01', periods=days)

# 価格データ (円)
prices = 2000 + np.cumsum(np.random.randn(days) * 30)
df = pd.DataFrame(data={'Close': prices}, index=dates)

# 予測データ (ノイズあり)
noise = np.random.normal(0, 40, days)
df['Predicted_Next_Close'] = df['Close'].shift(-1) + noise

# ==========================================
# 2. 戦略ロジック (株数決定)
# ==========================================
# 予測リターン
df['Predicted_Return'] = (df['Predicted_Next_Close'] - df['Close']) / df['Close']

# 目標株数 = リターン × 感度
df['Raw_Shares'] = df['Predicted_Return'] * GAIN_SHARES

# クリッピング (-MAX 〜 +MAX)
# 規格化するわけではない。MAXを超えたものをMAXに制限するだけ。
if ALLOW_SHORT:
    df['Target_Shares'] = df['Raw_Shares'].clip(lower=-MAX_SHARES, upper=MAX_SHARES)
else:
    df['Target_Shares'] = df['Raw_Shares'].clip(lower=0, upper=MAX_SHARES)

# 整数丸め
if INTEGER_MODE:
    df['Shares'] = df['Target_Shares'].round(0)
else:
    df['Shares'] = df['Target_Shares']

# ==========================================
# 3. 損益計算 (金額ベース)
# ==========================================
df['Price_Diff'] = df['Close'].diff()
df['Gross_PnL'] = df['Shares'].shift(1) * df['Price_Diff']

# コスト計算
df['Share_Change'] = df['Shares'].diff().abs().fillna(0)
if ENABLE_COST:
    df['Transaction_Cost'] = df['Share_Change'] * COST_PER_SHARE
else:
    df['Transaction_Cost'] = 0.0

df['Net_PnL'] = df['Gross_PnL'] - df['Transaction_Cost']
df['Cum_PnL'] = df['Net_PnL'].cumsum()

# ==========================================
# 4. 可視化 (ここを修正しました)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1, 2]})

# --- 上段: 実測値 vs 予測値 ---
ax1.plot(df.index, df['Close'], label='Actual Price', color='black', alpha=0.6)
# 予測値を1日ずらして「予測していた今日の価格」として表示
ax1.plot(df.index, df['Predicted_Next_Close'].shift(1), label='Predicted Price (Shifted)',
         color='orange', linestyle='--', alpha=0.8)

ax1.set_title('1. Price: Actual vs Predicted')
ax1.set_ylabel('Price (JPY)')
ax1.legend(loc='upper left')
ax1.grid(True)

# --- 中段: 保有株数 ---
ax2.plot(df.index, df['Shares'], label='Number of Shares', color='green', drawstyle='steps-post')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_title(f'2. Position Size (Max: {MAX_SHARES}, Gain: {GAIN_SHARES})')
ax2.set_ylabel('Shares')
ax2.grid(True)

# --- 下段: 累積損益 ---
cost_str = f"ON (@{COST_PER_SHARE}JPY)" if ENABLE_COST else "OFF"
ax3.plot(df.index, df['Cum_PnL'], label=f'Total Profit (Cost: {cost_str})', color='blue', linewidth=2)
ax3.axhline(0, color='red', linestyle='--', linewidth=1)
ax3.set_title('3. Cumulative Profit (JPY)')
ax3.set_ylabel('Profit (JPY)')
ax3.legend(loc='upper left')
ax3.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 結果サマリー
# ==========================================
print(f"--- Strategy Report ---")
print(f"総売買株数: {df['Share_Change'].sum():.0f} 株")
print(f"総手数料: {df['Transaction_Cost'].sum():.0f} 円")
print(f"最終損益: {df['Cum_PnL'].iloc[-1]:.0f} 円")

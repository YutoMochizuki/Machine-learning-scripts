import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

# ==========================================
# 1. データ準備（ダミーデータ生成）
# ※ 実際のご自身のデータを使用する場合はここを置き換えてください
# ==========================================
np.random.seed(42)
t = np.linspace(0, 100, 1000)
# サイン波 + トレンド + ノイズ
raw_data = np.sin(t) + t * 0.05 + np.random.normal(0, 0.1, 1000)

# DataFrame化 (ここでは1変数の時系列を想定していますが、多変量でも構造は同じです)
df = pd.DataFrame(raw_data, columns=['value'])

# ==========================================
# 2. 前処理: スケーリングとシーケンス化
# ==========================================
# LSTMはスケールに敏感なため、0-1に正規化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df)

# 時系列データをLSTM用の3次元形式 (Samples, TimeSteps, Features) に変換する関数
def create_dataset(dataset, look_back=10):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :]) # 過去 look_back 分の特徴量
        Y.append(dataset[i + look_back, 0])     # 次の時点のターゲット
    return np.array(X), np.array(Y)

# ハイパーパラメータ設定（仮）
LOOK_BACK = 20  # 過去20ステップを見て次を予測
X, y = create_dataset(data_scaled, look_back=LOOK_BACK)

# データを前半80%を学習、後半20%をテスト（最終検証用）に分割
train_size = int(len(X) * 0.8)
X_train_full, X_test = X[:train_size], X[train_size:]
y_train_full, y_test = y[:train_size], y[train_size:]

print(f"Input Shape: {X_train_full.shape}")
# 期待される出力例: (784, 20, 1) -> (サンプル数, タイムステップ, 特徴量数)

# ==========================================
# 3. モデル定義 (Sequential)
# ==========================================
def build_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential()

    # LSTM層: activationはデフォルト(tanh)推奨。GPU高速化が効き、勾配消失しにくい。
    # return_sequences=False は、最後のタイムステップの出力のみを次の層に渡す設定
    model.add(LSTM(units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout))

    # 表現力を高めるためのDense層 (ここでReLUを使う)
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(dropout))

    # 出力層: 回帰なのでlinear (恒等写像)
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ==========================================
# 4. クロスバリデーション (TimeSeriesSplit)
# ==========================================
tscv = TimeSeriesSplit(n_splits=4)
cv_scores = []

print("Starting Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full)):
    X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]

    # モデル構築
    model = build_lstm_model(input_shape=(X_tr.shape[1], X_tr.shape[2]))

    # EarlyStopping: 過学習を防ぐ
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 学習
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, callbacks=[es], verbose=0)

    # 評価
    pred = model.predict(X_val, verbose=0)
    score = r2_score(y_val, pred)
    cv_scores.append(score)
    print(f"Fold {fold+1}: R2 = {score:.4f}")

print(f"Mean CV R2: {np.mean(cv_scores):.4f}")

# ==========================================
# 5. 最終学習とテストデータ評価
# ==========================================
print("\nFinal Training...")
final_model = build_lstm_model(input_shape=(X_train_full.shape[1], X_train_full.shape[2]))
es_final = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

final_model.fit(X_train_full, y_train_full, epochs=100, batch_size=32, callbacks=[es_final], verbose=1)

# テストデータで予測
y_pred_scaled = final_model.predict(X_test)

# スケールを元に戻す (逆変換)
# y_test は1次元なので reshape 必要
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred_scaled)

# 最終スコア
final_r2 = r2_score(y_test_inv, y_pred_inv)
print(f"Final Test R2: {final_r2:.4f}")

# ==========================================
# 6. 結果の可視化
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual Data', alpha=0.7)
plt.plot(y_pred_inv, label='LSTM Prediction', alpha=0.7, linestyle='--')
plt.title(f'LSTM Time Series Prediction (R2: {final_r2:.3f})')
plt.legend()
plt.show()

# Financial Time Series Analysis & Forecasting Framework

## 1. Project Overview
本リポジトリは、金融時系列データ（主に価格、収益率、またはボラティリティ）の分析、特徴量エンジニアリング、および予測モデルの構築を行うためのPythonスクリプト群です。
計量経済学的手法（HAR）、機械学習（Decision Tree）、深層学習（LSTM）、および信号処理（FFT）を統合し、包括的な市場分析と予測精度の比較検証を行うことを目的としています。

## 2. Directory Structure & File Descriptions

### A. Exploratory Data Analysis (EDA) & Visualization
データの統計的性質を把握し、金融市場のスタイライズド・ファクト（Stylized Facts）を確認するためのモジュール群です。

* **`eda.py`**: 基礎的な探索的データ分析を行うスクリプト。基本統計量、分布の確認、欠損値処理などを含みます。
* **`ccf.py`**: Cross-Correlation Function（相互相関係数）の計算とプロットを行い、変数間のリード・ラグ関係（先行・遅行）を特定します。

### B. Feature Engineering & Unsupervised Learning
時系列データから有意なパターンを抽出し、モデリングのための特徴量を生成・選定します。

* **`fft.py`**: Fast Fourier Transform（高速フーリエ変換）。時系列データを周波数領域に変換し、周期性（Seasonality）やノイズ除去、スペクトル特徴量の抽出を行います。
* **`clustering.py`**: 一般的なクラスタリング手法（K-Means等）の実装。市場局面の分類などに使用。
* **`vola-clustering.py`**: ボラティリティ（変動率）に特化したクラスタリング。高ボラティリティ/低ボラティリティ局面のレジーム検知を目的とします。

### C. Predictive Modeling
予測タスクを実行するためのモデル群です。線形・非線形、時系列特化型のモデルを含みます。

* **`regression.py`**: 基本的な線形回帰モデル（OLS, Ridge, Lasso等）の実装。ベースラインとして機能します。
* **`har.py`**: Heterogeneous Autoregressive (HAR-RV) モデル。異なる時間スケール（日次、週次、月次）の成分を用いた、実現ボラティリティ予測の標準的モデルです。
* **`garch.py`**: GARCH (Generalized Autoregressive Conditional Heteroskedasticity) モデルの実装。時系列の「条件付き分散」をモデル化し、ボラティリティ・クラスタリングを捉える標準的な計量経済学アプローチです。
* **`decision-tree.py`**: 決定木（CART, Random Forest, XGBoost等を含む可能性あり）による非線形回帰または分類モデル。
* **`lstm.py`**: Long Short-Term Memory（長短期記憶）ネットワーク。時系列の長期依存性を学習するためのRNNベースの深層学習モデル。

### D. Validation & Evaluation
時系列データの順序性を保ったまま、モデルの汎化性能を評価するためのフレームワークです。

* **`regression_walkforward.py`**: ウォークフォワード検証（Walk-Forward Validation）の実装。スライディングウィンドウを用いて、実運用に近い形式でモデルのパフォーマンスを評価します。Look-ahead bias（未来の情報の漏洩）を防ぐために重要です。

## 3. Usage (Example)
典型的な分析ワークフローは以下の通りです。

1.  **EDA**: `eda.py` を実行し、データの健全性と分布を確認。
2.  **Feature Extraction**: `fft.py` や `ccf.py` で予測に有効な特徴量を生成。
3.  **Model Training**: `har.py` (ベースライン) と `lstm.py` (提案手法) をトレーニング。
4.  **Validation**: `regression_walkforward.py` を使用して、アウトオブサンプルでの予測精度（RMSE, MAE等）を比較。

# 鋳造品の欠陥検出システム - Part 1: セットアップと設定

## 1. ライブラリのインポート

このセクションでは、画像処理と欠陥検出に必要な主要なPythonライブラリをインポートします。

### 必要なライブラリ
- **OpenCV (`cv2`)**: 画像処理の基本機能、エッジ検出、円検出などの主要な画像処理操作を提供
- **NumPy (`np`)**: 画像データの数値計算と配列操作を行う
- **scikit-image (`skimage`)**: 高度な画像処理機能（特徴抽出、画像計測など）を提供
- **Matplotlib (`plt`)**: 画像の表示と可視化を行う
- **pandas (`pd`)**: データの構造化と分析を行う

### セットアップ手順
1. 必要なライブラリをインストール:
   ```python
   %pip install -r requirements.txt
   ```
   このコマンドは、`requirements.txt`に記載された全ての必要なライブラリをインストールします。

2. 必要なライブラリをインポート:
   ```python
   import os
   import cv2
   import numpy as np
   from skimage import feature, measure
   from skimage.morphology import skeletonize
   import matplotlib.pyplot as plt
   import matplotlib.patches as patches
   import pandas as pd
   ```

## 2. パラメータの設定

このセクションでは、画像処理パイプラインの様々な段階で使用される重要なパラメータを定義します。これらのパラメータは、システムの性能と精度に直接影響を与えます。

### ディレクトリとファイルパスの設定
```python
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"
processing_template_dir = os.path.join(input_data_dir, "processing_template")
```
- `input_data_dir`: 入力画像が格納されているディレクトリ
- `output_data_dir`: 処理結果を保存するディレクトリ
- `processing_template_dir`: テンプレート画像が格納されているディレクトリ

### テンプレートマッチングの設定
```python
right_processing_template_path = os.path.join(processing_template_dir, "right_processing.jpg")
left_processing_template_path = os.path.join(processing_template_dir, "left_processing.jpg")
template_match_threshold = 0.8
```
- テンプレートマッチングの閾値（0.8）は、マッチングの厳密さを制御
- 値が大きいほど、より正確なマッチングが要求される
- 範囲: 0.0 〜 1.0（1.0が完全一致）

### 欠陥検出パラメータ

#### 1. 円検出パラメータ
```python
circle_dp = 1.0
circle_min_dist = 20
circle_param1 = 50
circle_param2 = 30
circle_min_radius = 10
circle_max_radius = 100
```
- `circle_dp`: 画像分解能の逆比（1.0が標準）
- `circle_min_dist`: 検出される円の最小間隔（ピクセル）
- `circle_param1`: エッジ検出感度
- `circle_param2`: 円検出の閾値
- `circle_min/max_radius`: 検出する円の半径範囲

#### 2. エッジ検出パラメータ（大きな欠陥用）
```python
canny_kernel_size = (3, 3)
canny_sigma = 2.0
canny_min_threshold = 55
canny_max_threshold = 250
canny_merge_distance = 15
```
- カーネルサイズとシグマ値は、ノイズ除去の強度を制御
- 閾値は、エッジとして検出する輝度変化の強さを決定
- 統合距離は、近接したエッジの結合を制御

#### 3. 差分ガウシアン（DoG）フィルタパラメータ（小さな欠陥用）
```python
dog_ksize = 9
dog_sigma1 = 1.5
dog_sigma2 = 3.5
dog_merge_distance = 15
```
- カーネルサイズは、フィルタの範囲を決定
- シグマ値の差が、検出する特徴のスケールを決定
- 統合距離は、近接した検出結果の結合を制御

### パラメータ調整のガイドライン

1. **画質に応じた調整**
   - ノイズが多い画像: カーネルサイズとシグマ値を大きくする
   - コントラストが低い: 閾値を下げる
   - 解像度が異なる: スケールパラメータを調整

2. **欠陥サイズに応じた調整**
   - 大きな欠陥: Cannyパラメータを調整
   - 小さな欠陥: DoGパラメータを調整
   - サイズ範囲: `min_large_defect_size`と`max_large_defect_size`を調整

3. **誤検出の低減**
   - テンプレートマッチング閾値を上げる
   - コントラスト比と輝度差の閾値を調整
   - エッジ補完パラメータを微調整

### 性能モニタリング
パラメータ調整後は以下の点を確認:
1. 検出率（見落としが少ないか）
2. 誤検出率（過剰検出が少ないか）
3. 処理速度（実用的な範囲か）

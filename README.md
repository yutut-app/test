# No1_Detecting_defect_candidates.ipynb

## 概要
本ノートブックは、Keyenceカメラで撮影したワーク画像から鋳巣（欠陥）候補を検出するための画像処理を実装したものである。処理は以下の8つのステップで構成される：

1. ライブラリのインポート
2. パラメータの設定
3. データの読み込み
4. 加工領域の特定
5. Canny+DoGによる欠陥候補検出
6. エッジ補完
7. 欠陥候補のフィルタリング
8. 欠陥候補の画像とCSV保存

主な機能：
- 撮影画像の左右判定
- 加工領域のマスク生成
- 大小の欠陥検出（Cannyエッジ検出とDoGフィルタの組み合わせ）
- 検出結果の後処理とフィルタリング
- 検出結果の保存と可視化

## 1. ライブラリのインポート

本プログラムで使用する主要なライブラリは以下の通りである：

### 基本ライブラリ
```python
import os            # ファイル・ディレクトリ操作
import cv2           # OpenCV画像処理ライブラリ
import numpy as np   # 数値計算ライブラリ
```

### 画像処理専用ライブラリ
```python
from skimage import measure           # 領域計測
from skimage.morphology import skeletonize  # 細線化処理
```

### データ処理・可視化ライブラリ
```python
import matplotlib.pyplot as plt  # グラフ描画
import pandas as pd             # データフレーム操作
```

### 依存ライブラリのインストール
必要なライブラリは`requirements.txt`からインストールする：
```python
%pip install -r requirements.txt
```

requirements.txtには以下のライブラリが記載されている必要がある：
- opencv-python
- numpy
- scikit-image
- matplotlib
- pandas

これらのライブラリは画像処理、数値計算、データ分析、可視化などの機能を提供し、本プログラムの各処理ステップで使用される。特に：

1. OpenCV (cv2)
   - 画像の読み込み・保存
   - 基本的な画像処理
   - エッジ検出
   - 円検出

2. scikit-image
   - 高度な画像処理
   - 領域の特徴量計算
   - モルフォロジー演算

3. NumPy
   - 画像データの配列操作
   - 数値計算処理

4. Pandas
   - 検出結果のデータ管理
   - CSVファイルの入出力

5. Matplotlib
   - 処理結果の可視化
   - デバッグ用の画像表示

これらのライブラリを組み合わせることで、効率的な画像処理と結果の分析が可能となる。

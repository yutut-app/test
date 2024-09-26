以下の内容に基づいて、項目1〜3の部分を`.ipynb`形式で実装し、同時に`requirements.txt`ファイルを生成します。

まず、要求に基づいて、処理ステップごとのコードを作成します。

### 1. ライブラリのインポート
```python
# ライブラリのインポート
import os
import numpy as np
from skimage import io, filters, color, measure
from skimage.filters import threshold_otsu
from skimage.feature import canny
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt

# %pip install -r requirements.txt が実行された後に必要なライブラリがインポートされます。
```

### 2. パラメータの設定
```python
# パラメータの設定
input_dir = r"../data/input"  # 入力画像ディレクトリ
output_dir = r"../data/output"  # 出力画像ディレクトリ
ok_dir = os.path.join(input_dir, "OK")
ng_dir = os.path.join(input_dir, "NG")
output_ok_dir = os.path.join(output_dir, "OK")
output_ng_dir = os.path.join(output_dir, "NG")

# 処理対象のワークのサイズ、エッジ検出のパラメータなど
edge_detection_sigma = 1.0
min_defect_size = 8  # 最小の欠陥サイズ (px)
work_area_threshold = 0.5  # 閾値設定でワーク領域を検出する際のしきい値
```

### 3. データの読み込み
```python
# データの読み込み
def load_images_from_directory(directory):
    images = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = io.imread(img_path)
            images[filename] = img
    return images

# OK/NGディレクトリの画像読み込み
ok_images = load_images_from_directory(ok_dir)
ng_images = load_images_from_directory(ng_dir)

print(f"Loaded {len(ok_images)} OK images and {len(ng_images)} NG images.")
```

### 4. `requirements.txt`の生成
必要なライブラリを`requirements.txt`にリストアップします。

```plaintext
numpy
scikit-image
matplotlib
```

---

これで、項目1～3までの部分が実装されました。次に進める準備が整ったら、ワーク領域の検出や欠陥候補の検出、切り出しに進みます。これらの内容は次のステップで説明・実装できます。

これらのコードを元に.ipynbファイルを生成し、実行したい場合はお知らせください。

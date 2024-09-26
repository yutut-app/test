まず、以下のライブラリが必要であるため、`requirements.txt`を生成しました。内容は以下の通りです。

```
scikit-image
opencv-python
numpy
matplotlib
```

次に、各ステップの処理について`ipynb`形式で作成します。

### ステップ1: ライブラリのインポート
以下のコードを使い、ライブラリをインポートします。

```python
# Step 1: Import necessary libraries
import os
import numpy as np
import cv2
from skimage import io, filters, feature, color
import matplotlib.pyplot as plt
```

### ステップ2: パラメータの設定
以下のパラメータを設定します。主にディレクトリや画像のサイズ、欠陥の閾値などを定義します。

```python
# Step 2: Parameter settings
input_dir = r"../data/input"
output_dir = r"../data/output"
ok_dir = os.path.join(input_dir, "OK")
ng_dir = os.path.join(input_dir, "NG")
labels = ["label1", "label2", "label3"]  # label1: 鋳巣, label2: 凹み, label3: 亀裂
shape_suffix = "Shape"
normal_suffix = "Normal"
```

### ステップ3: データの読み込み
データを一つずつ処理するための読み込み方法を実装します。

```python
# Step 3: Data loading (one by one to simulate real-time processing)
def load_images(work_type="NG"):
    if work_type == "OK":
        image_dir = ok_dir
    else:
        image_dir = ng_dir
    
    # Load all images from the directory
    for label in labels:
        label_dir = os.path.join(image_dir, label)
        if os.path.exists(label_dir):
            for img_file in os.listdir(label_dir):
                if img_file.endswith(".jpg"):
                    origin_img_path = os.path.join(label_dir, img_file.replace(shape_suffix, normal_suffix))
                    shape_img_path = os.path.join(label_dir, img_file)
                    yield io.imread(origin_img_path), io.imread(shape_img_path), img_file

# Example to load images
for origin_img, shape_img, file_name in load_images("NG"):
    print(f"Processing file: {file_name}")
    # Here you can continue with further processing
    break  # Just load one image for demonstration
```

これらのステップで、まずキーエンスの前処理画像を読み込み、次に欠陥候補を検出する準備を整えます。

この後のステップで、元画像からワークのH面領域を検出し、欠陥候補の抽出と画像の切り出しを行いますが、まずここまでのステップでipynbファイルを作成してから次に進みます。

これらを元にして、`ipynb`ファイルを実行する準備ができています。次に進める準備ができましたらお知らせください。

---

これで、項目1～3までの部分が実装されました。次に進める準備が整ったら、ワーク領域の検出や欠陥候補の検出、切り出しに進みます。これらの内容は次のステップで説明・実装できます。

これらのコードを元に.ipynbファイルを生成し、実行したい場合はお知らせください。

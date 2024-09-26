以下に、項目4.1（二直化）を実装したコードを示します。元画像に対して二直化を行い、しきい値150を用いて`THRESH_BINARY`（二値化）を適用します。

### 追加ライブラリのインポートとパラメータ設定
二直化のために、`skimage`の追加モジュールをインポートし、しきい値やその他のパラメータを設定します。

### .ipynbの内容

#### 1. ライブラリのインポート
```python
# Import necessary libraries
%pip install -r requirements.txt

import os
import cv2
import numpy as np
from skimage import io, filters, measure
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_local
```

#### 2. パラメータの設定
```python
# Set parameters
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"
ng_labels = ['label1', 'label2', 'label3']  # label1: porosity, label2: dents, label3: cracks (亀裂)
work_right_dir = os.path.join(input_data_dir, "work_right")
work_left_dir = os.path.join(input_data_dir, "work_left")

# Binarization parameters
binarization_threshold = 150  # 二直化しきい値
```

#### 3. データの読み込み
```python
# Function to load image paths from the directory
def load_images_from_directory(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

# Load NG images from each label directory
ng_images_label1 = load_images_from_directory(os.path.join(input_data_dir, "NG", "label1"))
ng_images_label2 = load_images_from_directory(os.path.join(input_data_dir, "NG", "label2"))
ng_images_label3 = load_images_from_directory(os.path.join(input_data_dir, "NG", "label3"))

# Load OK images
ok_images = load_images_from_directory(os.path.join(input_data_dir, "OK"))
```

#### 4. 元画像からワークのH面領域を検出（省略）

#### 4.1 二直化（元画像のみ）
```python
# Function to perform binarization
def binarize_image(image, threshold_value):
    """
    入力画像をグレースケール化し、指定されたしきい値で二直化を行う関数
    """
    # グレースケールに変換
    gray_image = rgb2gray(image)
    
    # 二直化処理を実行 (しきい値THRESH_BINARYを使用)
    binary_image = gray_image > threshold_value / 255.0
    
    return binary_image

# Display binarized version of the first NG image (origin) from label1
if len(ng_images_label1) >= 1:
    # 最初の画像（元画像）を読み込む
    origin_image_path = ng_images_label1[0]
    origin_image = io.imread(origin_image_path)

    # 二直化処理を実行
    binarized_image = binarize_image(origin_image, binarization_threshold)

    # 二直化画像を表示
    plt.imshow(binarized_image, cmap='gray')
    plt.title(f"Binarized Image (Threshold: {binarization_threshold}) - {os.path.basename(origin_image_path)}")
    plt.axis('off')
    plt.show()
else:
    print("NG_label1の画像がありません。")
```

### 説明
1. **ライブラリのインポート**において、`skimage.color.rgb2gray`および`skimage.filters.threshold_otsu`を追加でインポートしました。
2. **パラメータの設定**では、二直化のしきい値（150）を`binarization_threshold`として設定しています。
3. **データの読み込み**部分は、前述の通りです。
4. **二直化処理**では、グレースケールに変換した後、指定したしきい値を使って二直化（`THRESH_BINARY`）を行います。二直化後の画像は、`binary_image`として保存され、`imshow`で表示されます。

このコードを実行すると、NG_label1の最初の元画像が二直化されて表示されます。

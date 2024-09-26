次のステップで、項目4.1（二直化処理）を追加し、元画像に対してしきい値を適用し、`matched_images`リストを更新します。また、更新された`matched_images`の最初のペアの画像を表示します。

### .ipynb構成

#### 1. ライブラリのインポート
```python
# ライブラリのインポート
%pip install -r requirements.txt

import os
import cv2
import numpy as np
from skimage import io, filters, feature, measure
import matplotlib.pyplot as plt
```

#### 2. パラメータの設定
```python
# パラメータの設定
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"
ng_labels = ['label1', 'label2', 'label3']  # label1: porosity, label2: dents, label3: cracks (亀裂)
work_right_dir = os.path.join(input_data_dir, "work_right")
work_left_dir = os.path.join(input_data_dir, "work_left")
threshold_value = 150  # 二直化のしきい値
```

#### 3. データの読み込み
```python
# NormalとShape画像をベースネームでマッチングして読み込む関数
def load_origin_keyence_images(directory):
    normal_images = {}
    shape_images = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "Normal" in file and file.endswith(".jpg"):
                base_name = file.replace("Normal", "")
                normal_images[base_name] = os.path.join(root, file)
            elif "Shape" in file and file.endswith(".jpg"):
                base_name = file.replace("Shape", "")
                shape_images[base_name] = os.path.join(root, file)
    
    matched_images = []
    for base_name in normal_images:
        if base_name in shape_images:
            matched_images.append((normal_images[base_name], shape_images[base_name]))
    return matched_images

# NGとOKのデータを読み込む
ng_images_label1 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label1"))
ok_images = load_origin_keyence_images(os.path.join(input_data_dir, "OK"))
```

#### 4. 元画像からワークのH面領域を検出
このステップではH面の検出が前提となる処理です。今回は説明のためにスキップしていますが、実際のコードには含まれるべきです。

#### 4.1 二直化処理
```python
# 二直化処理を元画像に適用し、matched_imagesを更新する
def binarize_images(matched_images, threshold):
    updated_images = []
    for normal_image_path, keyence_image_path in matched_images:
        # 元画像を読み込み
        normal_image = io.imread(normal_image_path)

        # 二直化処理を適用 (THRESH_BINARY_INV)
        _, binarized_image = cv2.threshold(normal_image, threshold, 255, cv2.THRESH_BINARY_INV)

        # 更新された画像リストに追加
        updated_images.append((binarized_image, keyence_image_path))

    return updated_images

# NG画像のペアに対して二直化処理を行う
matched_images = binarize_images(ng_images_label1, threshold_value)
```

#### 5. 最初のペアの画像を表示
```python
# 更新されたmatched_imagesの最初のペアを表示
if matched_images:
    binarized_image, keyence_image_path = matched_images[0]

    # Keyence画像を読み込み
    keyence_image = io.imread(keyence_image_path)

    # 二直化された元画像とKeyence画像を表示
    plt.figure(figsize=(10, 5))

    # 二直化された元画像
    plt.subplot(1, 2, 1)
    plt.imshow(binarized_image, cmap='gray')
    plt.title("Binarized Origin Image")
    plt.axis('off')

    # Keyence画像
    plt.subplot(1, 2, 2)
    plt.imshow(keyence_image, cmap='gray')
    plt.title(f"Keyence Image - {os.path.basename(keyence_image_path)}")
    plt.axis('off')

    plt.show()
else:
    print("No matched images found.")
```

### requirements.txt
```plaintext
opencv-python
scikit-image
matplotlib
numpy
```

### 説明
1. **ライブラリのインポート**：ここに追加項目はありませんが、`opencv-python`が二直化処理に必要です。
2. **パラメータの設定**：二直化しきい値`threshold_value`を追加しました。
3. **データの読み込み**：`Normal`と`Shape`画像のペアをベースネームでマッチングして読み込みます。
4. **二直化処理**：`cv2.threshold()`を使用して、元画像に二直化処理を適用します。処理結果は`matched_images`に更新されます。
5. **画像表示**：更新された`matched_images`の最初のペアを表示します。二直化された元画像とキーエンス前処理画像を並べて確認できます。

この処理により、リアルタイムに模擬して1ワークずつ処理し、メモリと処理時間を効率的に使用できる構成になっています。

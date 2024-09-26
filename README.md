以下に、ステップ4.2（ワーク接合部の削除）を含めた全体の処理を示します。このステップでは、テンプレートマッチングを使用してワークの左右を判別し、対応する方向から3730ピクセルで切り出します。

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
crop_width = 3730  # ワーク接合部を削除する際の見切り位置
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

#### 4.1 二直化処理
```python
# 二直化処理を元画像に適用し、matched_imagesを更新する
def binarize_images(matched_images, threshold):
    updated_images = []
    for normal_image_path, keyence_image_path in matched_images:
        # 元画像を読み込み
        normal_image = io.imread(normal_image_path)

        # 二直化処理を適用 (THRESH_BINARY)
        _, binarized_image = cv2.threshold(normal_image, threshold, 255, cv2.THRESH_BINARY)

        # 更新された画像リストに追加
        updated_images.append((binarized_image, keyence_image_path))

    return updated_images

# NG画像のペアに対して二直化処理を行う
matched_images = binarize_images(ng_images_label1, threshold_value)
```

#### 4.2 ワーク接合部の削除（テンプレートマッチングを使用して左右を判定）
```python
# ワーク接合部を削除する関数
def remove_joint_part(matched_images, work_right_template, work_left_template, crop_width):
    updated_images = []
    for binarized_image, keyence_image_path in matched_images:
        # Keyence画像の読み込み
        keyence_image = io.imread(keyence_image_path)

        # テンプレートマッチングを使用して右側か左側かを判定
        result_right = cv2.matchTemplate(binarized_image, work_right_template, cv2.TM_CCOEFF)
        result_left = cv2.matchTemplate(binarized_image, work_left_template, cv2.TM_CCOEFF)

        # 結果に基づき、右側か左側を判断
        min_val_right, max_val_right, min_loc_right, max_loc_right = cv2.minMaxLoc(result_right)
        min_val_left, max_val_left, min_loc_left, max_loc_left = cv2.minMaxLoc(result_left)

        if max_val_right > max_val_left:
            # 右側のワーク → 左から3730ピクセルで切り出す
            binarized_image_cropped = binarized_image[:, :crop_width]
            keyence_image_cropped = keyence_image[:, :crop_width]
        else:
            # 左側のワーク → 右から3730ピクセルで切り出す
            binarized_image_cropped = binarized_image[:, -crop_width:]
            keyence_image_cropped = keyence_image[:, -crop_width:]

        # 切り出した画像を更新
        updated_images.append((binarized_image_cropped, keyence_image_cropped))

    return updated_images

# ワーク右側と左側のテンプレート画像を読み込む
work_right_template = io.imread(os.path.join(work_right_dir, "template_right.jpg"), as_gray=True)
work_left_template = io.imread(os.path.join(work_left_dir, "template_left.jpg"), as_gray=True)

# ワーク接合部の削除処理を実行
updated_images = remove_joint_part(matched_images, work_right_template, work_left_template, crop_width)
```

#### 5. 最初のペアの画像を表示
```python
# 更新されたupdated_imagesの最初のペアを表示
if updated_images:
    binarized_image_cropped, keyence_image_cropped = updated_images[0]

    # 二直化された元画像とKeyence画像を表示
    plt.figure(figsize=(10, 5))

    # 二直化された元画像
    plt.subplot(1, 2, 1)
    plt.imshow(binarized_image_cropped, cmap='gray')
    plt.title("Cropped Binarized Origin Image")
    plt.axis('off')

    # Keyence画像
    plt.subplot(1, 2, 2)
    plt.imshow(keyence_image_cropped, cmap='gray')
    plt.title(f"Cropped Keyence Image")
    plt.axis('off')

    plt.show()
else:
    print("No updated images found.")
```

### requirements.txt
```plaintext
opencv-python
scikit-image
matplotlib
numpy
```

### 説明
1. **ライブラリのインポート**：ここには特に追加項目はありません。
2. **パラメータの設定**：接合部削除のための`crop_width`（3730ピクセル）や、テンプレート画像のパスを設定します。
3. **データの読み込み**：元画像とキーエンス画像のペアを読み込みます。
4. **二直化処理**：しきい値を使用して元画像を二直化します。
5. **ワーク接合部の削除**：テンプレートマッチングにより、ワークが右側か左側かを判定し、接合部を削除する処理を行います。
6. **画像表示**：処理後の二直化された元画像とキーエンス画像を表示します。

これで、ワークの左右判定に基づいた接合部の削除を行い、メモリ効率のよい処理を実現しています。

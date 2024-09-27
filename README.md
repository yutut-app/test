以下のコードは、指定された処理を実行するための.ipynbファイルで、元画像（Normal）とキーエンス前処理画像（Shape）のワーク接合部の削除を行い、テンプレートマッチングによってワークが左右どちらにあるかを判定します。さらに、`updated_images`リストを更新し、最初のペアの画像を表示します。

### .ipynb構成

#### 1. ライブラリのインポート
```python
# 必要なライブラリのインポート
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
ng_labels = ['label1', 'label2', 'label3']  # label1: 鋳巣, label2: 凹み, label3: 亀裂
work_right_dir = os.path.join(input_data_dir, "work_right")
work_left_dir = os.path.join(input_data_dir, "work_left")
crop_px = 1360  # ワークの接合部の切り出し量
```

#### 3. データの読み込み
```python
# Normal(元画像)とShape(キーエンス前処理画像)を一致させて読み込む関数
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
    
    # NormalとShapeが一致したペアを作成
    matched_images = []
    for base_name in normal_images:
        if base_name in shape_images:
            matched_images.append((normal_images[base_name], shape_images[base_name]))
    return matched_images

# NG画像とOK画像を読み込む
ng_images_label1 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label1"))
ng_images_label2 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label2"))
ng_images_label3 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label3"))
ok_images = load_origin_keyence_images(os.path.join(input_data_dir, "OK"))
```

#### 4. ワーク接合部の削除（テンプレートマッチング）
```python
# テンプレートマッチングでワークの右側・左側を判定し、接合部を削除する関数
def remove_joint_part(image, template_right, template_left):
    # テンプレートマッチングを実施して右側か左側かを判定
    result_right = cv2.matchTemplate(image, template_right, cv2.TM_CCOEFF)
    _, _, _, max_loc_right = cv2.minMaxLoc(result_right)
    
    result_left = cv2.matchTemplate(image, template_left, cv2.TM_CCOEFF)
    _, _, _, max_loc_left = cv2.minMaxLoc(result_left)

    # 右側と左側のマッチ結果を比較
    if max_loc_right[0] > max_loc_left[0]:  # 右側のテンプレートが優勢
        return image[:, crop_px:]  # 左から1360ピクセルを切り出す
    else:  # 左側のテンプレートが優勢
        return image[:, :-crop_px]  # 右から1360ピクセルを切り出す

# テンプレート画像を読み込む
template_right = io.imread(os.path.join(work_right_dir, "template_right.jpg"), as_gray=True)
template_left = io.imread(os.path.join(work_left_dir, "template_left.jpg"), as_gray=True)

# updated_imagesリストを初期化
updated_images = []
for origin_image_path, keyence_image_path in ng_images_label1 + ng_images_label2 + ng_images_label3 + ok_images:
    # 画像を読み込む
    origin_image = io.imread(origin_image_path, as_gray=True)
    keyence_image = io.imread(keyence_image_path, as_gray=True)

    # ワーク接合部を削除
    cropped_origin = remove_joint_part(origin_image, template_right, template_left)
    cropped_keyence = remove_joint_part(keyence_image, template_right, template_left)
    
    # updated_imagesに追加
    updated_images.append((cropped_origin, cropped_keyence))
```

#### 5. 更新された画像の表示
```python
# 最初のペアの更新された画像を表示
if updated_images:
    cropped_origin_image, cropped_keyence_image = updated_images[0]
    
    plt.figure(figsize=(10, 5))
    
    # 元画像の表示
    plt.subplot(1, 2, 1)
    plt.imshow(cropped_origin_image, cmap='gray')
    plt.title("Cropped Origin Image")
    plt.axis('off')
    
    # キーエンス前処理画像の表示
    plt.subplot(1, 2, 2)
    plt.imshow(cropped_keyence_image, cmap='gray')
    plt.title("Cropped Keyence Image")
    plt.axis('off')
    
    plt.show()
else:
    print("No images to display.")
```

### requirements.txt
```plaintext
opencv-python
scikit-image
matplotlib
numpy
```

### 説明
1. **ライブラリのインポート**では、`requirements.txt`から必要なライブラリをインポートしています。画像処理に`opencv`と`skimage`を使用します。
2. **パラメータの設定**で、画像ディレクトリや切り出し量を指定しています。
3. **データの読み込み**では、NormalとShapeの画像をペアにして読み込んでいます。
4. **ワーク接合部の削除**では、テンプレートマッチングによってワークが右側か左側かを判定し、指定した範囲を切り出します。
5. **画像の表示**では、接合部を削除した最初の画像ペアを表示します。

このコードは、効率を考慮しつつ少ないメモリで処理できるよう設計されています。

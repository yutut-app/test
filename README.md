以下の内容で、ワーク接合部の削除とペア画像の処理を行い、テンプレートマッチングで左右判別を加えた.ipynbファイルのコードを生成します。

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
ng_labels = ['label1', 'label2', 'label3']  # label1: 鋳巣, label2: 凹み, label3: 亀裂
template_dir = os.path.join(input_data_dir, "template")
right_template_path = os.path.join(template_dir, "right_keyence.jpg")
left_template_path = os.path.join(template_dir, "left_keyence.jpg")

crop_size = 1360  # ワーク接合部の削除に使用するピクセル数
```

#### 3. データの読み込み
```python
# ファイル名を基にorigin(元画像)とshape(キーエンス画像)をペアで読み込む関数
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
    
    # NormalとShapeのマッチングペアをリストで返す
    matched_images = []
    for base_name in normal_images:
        if base_name in shape_images:
            matched_images.append((normal_images[base_name], shape_images[base_name]))
    return matched_images

# NG画像をそれぞれのラベルから読み込み
ng_images_label1 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label1"))
ng_images_label2 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label2"))
ng_images_label3 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label3"))

# OK画像を読み込み
ok_images = load_origin_keyence_images(os.path.join(input_data_dir, "OK"))
```

#### 4. ワーク接合部の削除
```python
# テンプレートマッチングによって左右を判別し、接合部を削除する関数
def remove_joint_part(origin_image, keyence_image):
    # キーエンス画像を用いてテンプレートマッチングを実施
    keyence_img_cv = cv2.imread(keyence_image, cv2.IMREAD_UNCHANGED)
    
    # 右側テンプレートでマッチング
    template_right = cv2.imread(right_template_path, cv2.IMREAD_UNCHANGED)
    res_right = cv2.matchTemplate(keyence_img_cv, template_right, cv2.TM_CCOEFF)
    _, max_val_right, _, max_loc_right = cv2.minMaxLoc(res_right)
    
    # 左側テンプレートでマッチング
    template_left = cv2.imread(left_template_path, cv2.IMREAD_UNCHANGED)
    res_left = cv2.matchTemplate(keyence_img_cv, template_left, cv2.TM_CCOEFF)
    _, max_val_left, _, max_loc_left = cv2.minMaxLoc(res_left)

    # 右側が一致した場合
    if max_val_right > max_val_left:
        # 左側から1360ピクセルカット
        cropped_origin = origin_image[:, crop_size:]
        cropped_keyence = keyence_img_cv[:, crop_size:]
    else:
        # 右側から1360ピクセルカット
        cropped_origin = origin_image[:, :-crop_size]
        cropped_keyence = keyence_img_cv[:, :-crop_size]

    return cropped_origin, cropped_keyence

# ワーク接合部の削除を全てのNG画像とOK画像に対して実行
updated_images = []
for images_pair in ng_images_label1 + ng_images_label2 + ng_images_label3 + ok_images:
    origin_image_path, keyence_image_path = images_pair
    origin_image = io.imread(origin_image_path)
    keyence_image = io.imread(keyence_image_path)
    
    # ワーク接合部を削除
    cropped_origin, cropped_keyence = remove_joint_part(origin_image, keyence_image)
    
    # 更新された画像をリストに追加
    updated_images.append((cropped_origin, cropped_keyence))
```

#### 5. 更新された画像の表示
```python
# 更新された最初のペアの画像を表示
if updated_images:
    cropped_origin_image, cropped_keyence_image = updated_images[0]

    # 元画像 (cropped_origin_image) の表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cropped_origin_image, cmap='gray')
    plt.title("Cropped Origin Image")
    plt.axis('off')

    # キーエンス処理画像 (cropped_keyence_image) の表示
    plt.subplot(1, 2, 2)
    plt.imshow(cropped_keyence_image, cmap='gray')
    plt.title("Cropped Keyence Image")
    plt.axis('off')

    plt.show()
else:
    print("更新された画像がありません。")
```

### requirements.txt
```plaintext
opencv-python
scikit-image
matplotlib
numpy
```

### 説明
1. **ライブラリのインポート**では、`requirements.txt`に基づいて必要なライブラリをインポートしています。
2. **パラメータの設定**で、入力・出力ディレクトリやテンプレート画像のパス、切り出すサイズを定義しています。
3. **データの読み込み**で、`Normal`と`Shape`画像のペアをマッチングし、NGとOKの画像を読み込んでいます。
4. **ワーク接合部の削除**で、テンプレートマッチングによって画像が左右どちらのワークかを判別し、接合部を削除する処理を行っています。
5. **更新された画像の表示**では、最初のペア画像を表示しています。

これにより、接合部の削除が行われ、左右の判別に基づいた画像処理が実施されています。

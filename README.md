以下は、項目6.1「エッジ検出」を追加し、全ての`cropped_keyence_image`に対してエッジ検出を行い、`binarized_images`を更新するコードです。また、更新後の最初のペアの画像を表示します。

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
template_dir = os.path.join(input_data_dir, "template")
right_template_path = os.path.join(template_dir, "right_keyence.jpg")
left_template_path = os.path.join(template_dir, "left_keyence.jpg")
crop_width = 1360  # ワーク接合部を削除するための幅
threshold_value = 150  # 二直化しきい値
kernel_size = (5, 5)  # カーネルサイズ
iterations_open = 3  # 膨張処理の繰り返し回数
iterations_close = 20  # 収縮処理の繰り返し回数
gaussian_kernel_size = (7, 7)  # ガウシアンブラーのカーネルサイズ
canny_min_threshold = 50  # エッジ検出の最小しきい値
canny_max_threshold = 150  # エッジ検出の最大しきい値
sigma = 5  # ガウシアンブラーの標準偏差
```

#### 3. データの読み込み
```python
# 同じワークの元画像とキーエンス画像をペアで読み込む関数
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

# NG画像とOK画像をそれぞれ読み込む
ng_images_label1 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label1"))
ng_images_label2 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label2"))
ng_images_label3 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label3"))
ok_images = load_origin_keyence_images(os.path.join(input_data_dir, "OK"))
```

#### 4. ワーク接合部の削除
```python
# テンプレートマッチングでワークの左右を判定
def template_matching(image, template_path):
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc

# ワークの接合部を削除する関数
def remove_joint_part(image_path, keyence_image_path):
    # 画像の読み込み
    image = io.imread(image_path)
    keyence_image = io.imread(keyence_image_path)
    
    # テンプレートマッチングによる左右判定
    right_val, _ = template_matching(keyence_image, right_template_path)
    left_val, _ = template_matching(keyence_image, left_template_path)
    
    # ワークの右側か左側かを判定し、接合部を削除
    if right_val > left_val:
        # ワークの右側：左から1360pxカット
        cropped_image = image[:, crop_width:]
        cropped_keyence_image = keyence_image[:, crop_width:]
    else:
        # ワークの左側：右から1360pxカット
        cropped_image = image[:, :-crop_width]
        cropped_keyence_image = keyence_image[:, :-crop_width]
    
    return cropped_image, cropped_keyence_image

# 全ての画像ペアに対して接合部を削除し、新しいリストを作成
def process_images(image_pairs):
    updated_images = []
    for origin_image_path, keyence_image_path in image_pairs:
        cropped_image, cropped_keyence_image = remove_joint_part(origin_image_path, keyence_image_path)
        updated_images.append((cropped_image, cropped_keyence_image))
    return updated_images

# NGとOK画像に対して接合部削除を実行
updated_ng_images_label1 = process_images(ng_images_label1)
updated_ng_images_label2 = process_images(ng_images_label2)
updated_ng_images_label3 = process_images(ng_images_label3)
updated_ok_images = process_images(ok_images)
```

#### 5.1 二直化によるマスクの作成
```python
# 二直化とマスク作成
def binarize_image(image):
    # 画像がすでにグレースケールかどうか確認
    if len(image.shape) == 3:
        # グレースケールに変換
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # 二直化処理（THRESH_BINARY_INVを使用）
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # カーネル作成と膨張・収縮処理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
    
    return binary_image

# 全てのcropped_imageに対して二直化を実行し、新しいリストを作成
def binarize_images(image_pairs):
    binarized_images = []
    for cropped_image, cropped_keyence_image in image_pairs:
        binarized_image = binarize_image(cropped_image)
        binarized_images.append((binarized_image, cropped_keyence_image))
    return binarized_images

# NGとOK画像に対して二直化を実行
binarized_ng_images_label1 = binarize_images(updated_ng_images_label1)
binarized_ng_images_label2 = binarize_images(updated_ng_images_label2)
binarized_ng_images_label3 = binarize_images(updated_ng_images_label3)
binarized_ok_images = binarize_images(updated_ok_images)
```

#### 6.1 エッジ検出
```python
# エッジ検出を行う関数
def detect_edges(cropped_keyence_image, binarized_image):
    # H面のマスクを適用して背景を除去
    masked_image = cv2.bitwise_and(cropped_keyence_image, cropped_keyence_image, mask=binarized_image)
    
    # ガウシアンブラーを適用
    blurred_image = cv2.GaussianBlur(masked_image, gaussian_kernel_size, sigma)
    
    # エッジ検出
    edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
    
    return edges

# 全てのcropped_keyence_imageに対してエッジ検出を実行し、新しいリストを作成
def detect_edges_in_images(binarized_images):
    edged_images = []
    for binarized_image, cropped_keyence_image in binarized_images:
        edges = detect_edges(cropped_keyence_image, binarized_image)
        edged_images.append((binarized_image, edges))
    return edged_images

# NGとOK画像に対してエッジ検出を実行
edged_ng_images_label1 = detect_edges_in_images(binarized_ng_images_label1)
edged_ng_images_label2 = detect_edges_in_images(binarized_ng_images_label2)
edged_ng_images_label3 = detect_edges_in_images(binarized_ng_images_label3)
edged_ok_images = detect_edges_in_images(binarized_ok_images)
```

#### 7. 更新した画像ペアの表示
```python
# 更新されたNG_label1の最初の画像ペアを表示
if edged_ng_images_label1:
    binarized_image, edge_image = edged_ng_images_label1[

    # 更新されたNG_label1の最初の画像ペアを表示
    binarized_image, edge_image = edged_ng_images_label1[0]
    
    # 二直化後の画像の表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(binarized_image, cmap='gray')
    plt.title("Binarized Image")
    plt.axis('off')

    # エッジ検出後の画像の表示
    plt.subplot(1, 2, 2)
    plt.imshow(edge_image, cmap='gray')
    plt.title("Edge Detection Image")
    plt.axis('off')

    plt.show()
else:
    print("No images found after edge detection.")
```

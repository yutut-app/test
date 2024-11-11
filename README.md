申し訳ありませんが、このレスポンスは長くなりすぎるため、分割してお送りする必要があります。まず最初の部分をお送りします。

```python
# 1. ライブラリのインポート

import os
import cv2
import numpy as np
from skimage import io, filters, feature, measure
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import pandas as pd
```

```python
# 2. パラメータの設定

# ディレクトリとファイルパス
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"
template_dir = os.path.join(input_data_dir, "template")
right_template_path = os.path.join(template_dir, "right_keyence.jpg")
left_template_path = os.path.join(template_dir, "left_keyence.jpg")

# ラベル定義
ng_labels = ['label1', 'label2', 'label3']  # label1: 鋳巣, label2: 凹み, label3: 亀裂

# 画像処理パラメータ
crop_width = 1360  # ワーク接合部を削除するための幅
threshold_value = 150  # 二値化しきい値
kernel_size = (5, 5)  # カーネルサイズ
iterations_open = 3  # 膨張処理の繰り返し回数
iterations_close = 20  # 収縮処理の繰り返し回数

# DoGフィルタのパラメータ
dog_ksize = 5  # DoGフィルタのカーネルサイズ
dog_sigma1 = 1.0  # 1つ目のガウシアンフィルタのシグマ
dog_sigma2 = 2.0  # 2つ目のガウシアンフィルタのシグマ

# 動的閾値処理のパラメータ
dynamic_ksize = 11  # 局所領域のサイズ
dynamic_c = 2  # 閾値調整用定数
dynamic_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C  # 適応的閾値処理の方法

# エッジ補完のパラメータ
edge_kernel_size = (3, 3)  # エッジ補完のカーネルサイズ
edge_open_iterations = 2   # ノイズ削除の繰り返し回数
edge_close_iterations = 2  # エッジ補完の繰り返し回数

# マスクエッジ検出のパラメータ
mask_edge_min_threshold = 100
mask_edge_max_threshold = 200
mask_edge_margin = 5  # マスクエッジの余裕幅（ピクセル単位）

# 欠陥サイズパラメータ
min_defect_size = 5  # 最小欠陥サイズ（0.5mm = 5px）
max_defect_size = 100  # 最大欠陥サイズ（10mm = 100px）

# 欠陥候補の保存パラメータ
enlargement_factor = 10  # 欠陥候補画像の拡大倍率
```

```python
# 3. データの読み込み

def load_origin_keyence_images(directory):
    normal_images = {}
    shape_images = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "Normal" in file and file.endswith(".jpg"):
                base_name = file.replace("Normal", "")
                normal_images[base_name] = (os.path.join(root, file), file)
            elif "Shape" in file and file.endswith(".jpg"):
                base_name = file.replace("Shape", "")
                shape_images[base_name] = (os.path.join(root, file), file)
    
    matched_images = []
    for base_name in normal_images:
        if base_name in shape_images:
            matched_images.append((normal_images[base_name][0], 
                                 shape_images[base_name][0], 
                                 shape_images[base_name][1]))
    return matched_images

# NG画像とOK画像をそれぞれ読み込む
ng_images_label1 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label1"))
ng_images_label2 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label2"))
ng_images_label3 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label3"))
ok_images = load_origin_keyence_images(os.path.join(input_data_dir, "OK"))
```

続きは次のメッセージで送ります。コードが長いため、分割して送信させていただきます。

# 6. DoGフィルタと動的閾値処理による欠陥検出

def difference_of_gaussian(img, ksize, sigma1, sigma2):
    # ガウシアンフィルタ適用
    gaussian_1 = cv2.GaussianBlur(img, (ksize, ksize), sigma1)
    gaussian_2 = cv2.GaussianBlur(img, (ksize, ksize), sigma2)
    
    # 2種のガウシアンフィルタ適用画像の差分
    dog = gaussian_1 - gaussian_2
    
    return dog

def dynamic_threshold(img, ksize, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, c=2):
    # 適応的閾値処理
    binary = cv2.adaptiveThreshold(img, 255, method, cv2.THRESH_BINARY_INV, ksize, c)
    return binary

def detect_defects_dog_dynamic(cropped_keyence_image, binarized_image):
    # DoGフィルタ適用
    dog_result = difference_of_gaussian(cropped_keyence_image, dog_ksize, dog_sigma1, dog_sigma2)
    
    # DoG結果を8ビット unsigned int に変換
    dog_result = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 動的閾値処理
    dynamic_result = dynamic_threshold(cropped_keyence_image, dynamic_ksize, dynamic_method, dynamic_c)
    
    # DoGと動的閾値の結果を組み合わせる
    combined_result = cv2.bitwise_and(dog_result, dynamic_result)
    
    # マスク適用
    masked_result = cv2.bitwise_and(combined_result, combined_result, mask=binarized_image)
    
    return masked_result

def process_images_for_defect_detection(binarized_images):
    processed_images = []
    for binarized_image, cropped_keyence_image, original_filename in binarized_images:
        defect_image = detect_defects_dog_dynamic(cropped_keyence_image, binarized_image)
        processed_images.append((binarized_image, defect_image, original_filename))
    return processed_images

# NGとOK画像に対して欠陥検出を実行
processed_ng_images_label1 = process_images_for_defect_detection(binarized_ng_images_label1)
processed_ng_images_label2 = process_images_for_defect_detection(binarized_ng_images_label2)
processed_ng_images_label3 = process_images_for_defect_detection(binarized_ng_images_label3)
processed_ok_images = process_images_for_defect_detection(binarized_ok_images)

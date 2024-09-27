以下は、ステップ5（キーエンス前処理画像から欠陥候補を検出）、ステップ6（画像の切り出し）、ステップ7（画像の保存）を追加した.ipynbの全体構成です。すべてのキーエンス前処理画像に対して欠陥候補を検出し、H面領域を基に画像を切り出して保存します。

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
template_dir = os.path.join(input_data_dir, "template")  # テンプレートディレクトリ
right_template = os.path.join(template_dir, "right_keyence.jpg")
left_template = os.path.join(template_dir, "left_keyence.jpg")
ng_labels = ['label1', 'label2', 'label3']  # label1: porosity, label2: dents, label3: cracks (亀裂)
threshold_value = 150  # 二直化のしきい値
crop_offset = 1360  # ワーク接合部の削除時のオフセット値
gaussian_kernel_size = (7, 7)  # ガウシアンブラーのカーネルサイズ
sigma_value = 5  # ガウシアンブラーのσ
edge_threshold_min = 50  # エッジ検出の最小閾値
edge_threshold_max = 150  # エッジ検出の最大閾値
min_defect_diameter = 5  # 欠陥の最小直径 (px)
max_defect_diameter = 100  # 欠陥の最大直径 (px)
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

H面検出は前提とする処理のため、ここでは省略。

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

        # H面が白色になるように処理する
        updated_images.append((binarized_image, keyence_image_path))

    return updated_images

# NG画像のペアに対して二直化処理を行う
matched_images = binarize_images(ng_images_label1, threshold_value)
```

#### 4.2 ワーク接合部の削除
```python
# テンプレートマッチングによってワークの左右を判定し、二直化画像とキーエンス画像の両方の接合部を削除する
def remove_joint_area(matched_images, right_template, left_template):
    updated_images = []
    
    # テンプレート画像の読み込み
    right_template_img = cv2.imread(right_template, cv2.IMREAD_GRAYSCALE)
    left_template_img = cv2.imread(left_template, cv2.IMREAD_GRAYSCALE)

    for binarized_image, keyence_image_path in matched_images:
        # Keyence画像の読み込み
        keyence_image = io.imread(keyence_image_path, as_gray=True)

        # テンプレートマッチングを実行し、どちら側かを判定
        right_res = cv2.matchTemplate(keyence_image, right_template_img, cv2.TM_CCOEFF)
        left_res = cv2.matchTemplate(keyence_image, left_template_img, cv2.TM_CCOEFF)

        # 最大値を持つマッチング結果に基づいて左右を判定
        right_max_val = cv2.minMaxLoc(right_res)[1]
        left_max_val = cv2.minMaxLoc(left_res)[1]

        # ワークが右側なら左から1360ピクセル見切る、左側なら右から1360ピクセル見切る
        if right_max_val > left_max_val:
            cropped_binarized_image = binarized_image[:, crop_offset:]
            cropped_keyence_image = keyence_image[:, crop_offset:]
        else:
            cropped_binarized_image = binarized_image[:, :-crop_offset]
            cropped_keyence_image = keyence_image[:, :-crop_offset]

        updated_images.append((cropped_binarized_image, cropped_keyence_image))

    return updated_images

# ワーク接合部を削除
updated_images = remove_joint_area(matched_images, right_template, left_template)
```

#### 5. キーエンス前処理画像から欠陥候補を検出
```python
# H面(白部分)に基づいて欠陥候補を検出する
def detect_defects(cropped_binarized_image, cropped_keyence_image, gaussian_kernel_size, sigma_value, edge_threshold_min, edge_threshold_max, min_defect_diameter, max_defect_diameter):
    # ガウシアンブラーを適用
    blurred_image = cv2.GaussianBlur(cropped_keyence_image, gaussian_kernel_size, sigma_value)

    # エッジ検出を実行
    edges = feature.canny(blurred_image, sigma=sigma_value, low_threshold=edge_threshold_min / 255, high_threshold=edge_threshold_max / 255)

    # H面の白部分の領域を特定し、欠陥候補を検出
    labels = measure.label(cropped_binarized_image, connectivity=2)
    regions = measure.regionprops(labels)

    defect_candidates = []

    for region in regions:
        # 欠陥候補のサイズが基準内か確認 (φ0.5mm以下、φ10mm以上は除外)
        if min_defect_diameter <= region.equivalent_diameter <= max_defect_diameter:
            defect_candidates.append(region.centroid)  # 欠陥候補の中心を取得

    return defect_candidates
```

#### 6. 画像の切り出し
```python
# 欠陥候補を中心に10px * 10pxの正方形を切り出し
def crop_defect_regions(cropped_keyence_image, defect_candidates, output_dir):
    for i, (y, x) in enumerate(defect_candidates):
        # 10px * 10pxで切り出す
        cropped_region = cropped_keyence_image[int(y)-5:int(y)+5, int(x)-5:int(x)+5]
        
        # ワークごとのディレクトリを作成して保存
        defect_dir = os.path.join(output_dir, f"defect_{i}")
        os.makedirs(defect_dir, exist_ok=True)
        
        # 画像の保存
        output_path = os.path.join(defect_dir, f"defect_{i}.jpg")
        io.imsave(output_path, cropped_region)
```

#### 7. 画像の保存
```python
# 欠陥候補の検出、画像の切り出しと保存を行う
for binarized_image, keyence_image in updated_images:
    defect_candidates = detect_defects(binarized_image, keyence_image, gaussian_kernel_size, sigma_value, edge_threshold_min, edge_threshold_max, min_defect_diameter, max_defect_diameter)
    crop_defect_regions(keyence_image, defect_candidates, output_data_dir)
```

### 説明
1. **ライブラリのインポート**：必要なライブラリをすべてインポートしています。
2. **パラメータの設定**：欠陥候

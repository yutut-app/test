以下は、項目5「キーエンス前処理画像から欠陥候補を検出」を追加し、`cropped_keyence_image`で欠陥候補の中心を検出する処理を行うコードです。

### .ipynb構成

#### 1. ライブラリのインポート
```python
# ライブラリのインポート
%pip install -r requirements.txt

import os
import cv2
import numpy as np
from skimage import io, feature, filters, measure
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
sigma = 5  # ガウシアンブラーのsigma
edge_min_val = 50  # エッジ検出の最小値
edge_max_val = 150  # エッジ検出の最大値
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

省略（H面の検出）。

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
# H面領域で欠陥候補を検出する処理
def detect_defects(updated_images, gaussian_kernel_size, sigma, edge_min_val, edge_max_val):
    defect_candidates = []

    for cropped_binarized_image, cropped_keyence_image in updated_images:
        # ガウシアンブラーを適用してノイズを低減
        blurred_keyence_image = cv2.GaussianBlur(cropped_keyence_image, gaussian_kernel_size, sigma)
        
        # エッジ検出を実行
        edges = feature.canny(blurred_keyence_image, sigma=sigma, low_threshold=edge_min_val / 255, high_threshold=edge_max_val / 255)

        # H面(白部分)と対応させ、欠陥候補を検出
        labeled_binarized = measure.label(cropped_binarized_image)
        regions = measure.regionprops(labeled_binarized)

        for region in regions:
            # H面にある領域に限定
            if np.any(edges[region.coords[:, 0], region.coords[:, 1]]):
                defect_candidates.append(region.centroid)  # 欠陥候補の中心を記録

    return defect_candidates

# 欠陥候補を検出
defect_candidates = detect_defects(updated_images, gaussian_kernel_size, sigma, edge_min_val, edge_max_val)

# 欠陥候補を確認
print("Detected defect candidates (centroids):", defect_candidates)
```

#### 6. 画像の切り出し

次の項目で処理します。

#### 7. 画像の保存

次の項目で処理します。

### 説明
1. **ライブラリのインポート**：OpenCVや`skimage`を使用して画像処理とエッジ検出を行います。
2. **パラメータの設定**：エッジ検出やガウシアンブラーのパラメータを設定します。
3. **データの読み込み**：`Normal`と`Shape`画像のペアをマッチングして読み込みます。
4. **二直化処理とワーク接合部の削除**：二直化画像とキーエンス画像の両方に対して、ワーク接合部を削除する処理を行います。
5. **欠陥候補の検出**：`cropped_binarized_image`でH面（白領域）を特定し、`cropped_keyence_image`に対してガウシアンブラーとエッジ検出を適用して欠陥候補の中心（`centroid`）を検出します。

この構成で、欠陥候補を検出する準備が整いました。

はい、ご要望に沿って改良したコードと詳細な説明を提供いたします。全ての項目を省略せずに説明します。

# 目次

- [1. ライブラリのインポート](#1-ライブラリのインポート)
- [2. パラメータの設定](#2-パラメータの設定)
- [3. データの読み込み](#3-データの読み込み)
- [4. ワーク接合部の削除](#4-ワーク接合部の削除)
- [5. 二値化によるマスクの作成](#5-二値化によるマスクの作成)
- [6. エッジ検出とテクスチャ検出の改良](#6-エッジ検出とテクスチャ検出の改良)
- [7. エッジの補完と欠陥候補の中心座標の取得](#7-エッジの補完と欠陥候補の中心座標の取得)
- [8. 欠陥候補のフィルタリング](#8-欠陥候補のフィルタリング)
- [9. 欠陥候補の画像の保存](#9-欠陥候補の画像の保存)

# 1. ライブラリのインポート

```python
import os
import cv2
import numpy as np
from skimage import io, filters, feature, measure
import matplotlib.pyplot as plt
import pandas as pd
```

この部分では、プロジェクトで必要なライブラリをインポートしています。

- `os`: ファイルやディレクトリの操作に使用します。
- `cv2`: OpenCVライブラリで、画像処理の主要な機能を提供します。
- `numpy`: 数値計算や配列操作に使用します。
- `skimage`: 画像処理のための追加機能を提供します。
- `matplotlib`: グラフや画像の表示に使用します。
- `pandas`: データ分析や操作のためのライブラリです。CSVファイルの作成に使用します。

# 2. パラメータの設定

```python
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"
ng_labels = ['label1', 'label2', 'label3']  # label1: 鋳巣, label2: 凹み, label3: 亀裂
template_dir = os.path.join(input_data_dir, "template")
right_template_path = os.path.join(template_dir, "right_keyence.jpg")
left_template_path = os.path.join(template_dir, "left_keyence.jpg")
crop_width = 1360  # ワーク接合部を削除するための幅
threshold_value = 150  # 二値化しきい値
kernel_size = (5, 5)  # カーネルサイズ
iterations_open = 3  # 膨張処理の繰り返し回数
iterations_close = 20  # 収縮処理の繰り返し回数
gaussian_kernel_size = (7, 7)  # ガウシアンブラーのカーネルサイズ
canny_min_threshold = 30  # エッジ検出の最小しきい値
canny_max_threshold = 120  # エッジ検出の最大しきい値
sigma = 3  # ガウシアンブラーの標準偏差
min_defect_size = 5  # 最小欠陥サイズ（0.5mm = 5px）
max_defect_size = 100  # 最大欠陥サイズ（10mm = 100px）
texture_threshold = 15  # テクスチャの変化を検出するためのしきい値
bounding_box_size = 200  # バウンディングボックスのサイズ
crop_size = 100  # 欠陥候補切り出しサイズ
enlarge_factor = 10  # 拡大倍率
```

これらのパラメータは、画像処理や欠陥検出の各段階で使用されます。必要に応じて調整することで、処理の精度や感度を制御できます。

# 3. データの読み込み

```python
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

ng_images_label1 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label1"))
ng_images_label2 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label2"))
ng_images_label3 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label3"))
ok_images = load_origin_keyence_images(os.path.join(input_data_dir, "OK"))
```

この関数は、指定されたディレクトリから元画像とキーエンス処理済み画像のペアを読み込みます。NGとOK画像それぞれに対してこの関数を適用し、画像ペアのリストを作成します。

# 4. ワーク接合部の削除

```python
def template_matching(image, template_path):
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc

def remove_joint_part(image_path, keyence_image_path):
    image = io.imread(image_path)
    keyence_image = io.imread(keyence_image_path)
    
    right_val, _ = template_matching(keyence_image, right_template_path)
    left_val, _ = template_matching(keyence_image, left_template_path)
    
    if right_val > left_val:
        cropped_image = image[:, crop_width:]
        cropped_keyence_image = keyence_image[:, crop_width:]
    else:
        cropped_image = image[:, :-crop_width]
        cropped_keyence_image = keyence_image[:, :-crop_width]
    
    return cropped_image, cropped_keyence_image

def process_images(image_pairs):
    updated_images = []
    for origin_image_path, keyence_image_path in image_pairs:
        cropped_image, cropped_keyence_image = remove_joint_part(origin_image_path, keyence_image_path)
        updated_images.append((cropped_image, cropped_keyence_image))
    return updated_images

updated_ng_images_label1 = process_images(ng_images_label1)
updated_ng_images_label2 = process_images(ng_images_label2)
updated_ng_images_label3 = process_images(ng_images_label3)
updated_ok_images = process_images(ok_images)
```

これらの関数は、テンプレートマッチングを使用してワークの左右を判定し、接合部を削除します。全ての画像ペアに対してこの処理を適用します。

# 5. 二値化によるマスクの作成

```python
def binarize_image(image):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
    
    return binary_image

def binarize_images(image_pairs):
    binarized_images = []
    for cropped_image, cropped_keyence_image in image_pairs:
        binarized_image = binarize_image(cropped_image)
        binarized_images.append((binarized_image, cropped_keyence_image))
    return binarized_images

binarized_ng_images_label1 = binarize_images(updated_ng_images_label1)
binarized_ng_images_label2 = binarize_images(updated_ng_images_label2)
binarized_ng_images_label3 = binarize_images(updated_ng_images_label3)
binarized_ok_images = binarize_images(updated_ok_images)
```

これらの関数は、画像を二値化し、モルフォロジー演算を適用してマスクを生成します。全ての画像ペアに対してこの処理を適用します。

# 6. エッジ検出とテクスチャ検出の改良

```python
def detect_edges_and_texture(cropped_keyence_image, binarized_image):
    masked_image = cv2.bitwise_and(cropped_keyence_image, cropped_keyence_image, mask=binarized_image)
    blurred_image = cv2.GaussianBlur(masked_image, gaussian_kernel_size, sigma)
    edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    laplacian_edges = np.uint8(abs_laplacian > texture_threshold) * 255
    combined_edges = cv2.bitwise_or(edges, laplacian_edges)
    return combined_edges

def process_edges(image_pairs):
    processed_images = []
    for binarized_image, cropped_keyence_image in image_pairs:
        edge_image = detect_edges_and_texture(cropped_keyence_image, binarized_image)
        processed_images.append((binarized_image, edge_image))
    return processed_images

edged_ng_images_label1 = process_edges(binarized_ng_images_label1)
edged_ng_images_label2 = process_edges(binarized_ng_images_label2)
edged_ng_images_label3 = process_edges(binarized_ng_images_label3)
edged_ok_images = process_edges(binarized_ok_images)
```

これらの関数は、エッジ検出とテクスチャ検出を組み合わせて欠陥を検出します。Cannyエッジ検出とラプラシアンフィルタを使用しています。

# 7. エッジの補完と欠陥候補の中心座標の取得

```python
def complete_edges(edge_image):
    kernel = np.ones((3,3), np.uint8)
    completed_edges = cv2.dilate(edge_image, kernel, iterations=1)
    completed_edges = cv2.erode(completed_edges, kernel, iterations=1)
    return completed_edges

def get_defect_candidates(edge_image):
    labels = measure.label(edge_image)
    properties = measure.regionprops(labels)
    
    candidates = []
    for prop in properties:
        candidates.append({
            'centroid': prop.centroid,
            'area': prop.area,
            'bbox': prop.bbox,
            'eccentricity': prop.eccentricity,
            'equivalent_diameter': prop.equivalent_diameter,
            'euler_number': prop.euler_number,
            'extent': prop.extent,
            'filled_area': prop.filled_area,
            'major_axis_length': prop.major_axis_length,
            'minor_axis_length': prop.minor_axis_length,
            'orientation': prop.orientation,
            'perimeter': prop.perimeter,
            'solidity': prop.solidity
        })
    
    return candidates

def visualize_labeled_image(image, candidates):
    labeled_image = image.copy()
    for i, candidate in enumerate(candidates):
        y, x = map(int, candidate['centroid'])
        cv2.circle(labeled_image, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(labeled_image, f"{i}", (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    plt.imshow(labeled_image)
    plt.title("Labeled Defect Candidates")
    plt.axis('off')
    plt.show()

def process_and_label_images(image_pairs):
    processed_images = []
    for binarized_image, edge_image in image_pairs:
        completed_edges = complete_edges(edge_image)
        candidates = get_defect_candidates(completed_edges)
        visualize_labeled_image(edge_image, candidates)
        processed_images.append((binarized_image, completed_edges, candidates))
    return processed_images

labeled_ng_images_label1 = process_and_label_images(edged_ng_images_label1)
labeled_ng_images_label2 = process_and_label_images(edged_ng_images_label2)
labeled_ng_images_label3 = process_and_label_images(edged_ng_images_label3)
labeled_ok_images = process_and_label_images(edged_ok_images)
```

これらの関数は、エッジを補完し、欠陥候補の特徴量を抽出します。また、ラベリングした画像を可視化します。

# 8. 欠陥候補のフィルタリング

```python
def visualize_mask_edges(mask):
    edges = cv2.Canny(mask, 100, 200)
    plt.imshow(edges, cmap='gray')
    plt.title("Mask Edges")
    plt.axis('off')
    plt.show()

def filter_candidates(candidates, mask, min_size, max_size):
    mask_edges = cv2.Canny(mask, 100, 200)
    filtered_candidates = []
    for candidate in candidates:
        y, x = map(int, candidate['centroid'])
        if min_size <= candidate['area'] <= max_size:
            if mask_edges[y, x] == 0:  # エッジ上にないか確認
                filtered_candidates.append(candidate)
    return filtered_candidates

def draw_bounding_boxes(image, candidates):
    result_image = image.copy()
    for i, candidate in enumerate(candidates):
        y, x = map(int, candidate['centroid'])
        top_left = (max(0, x - bounding_box_size // 2), max(0, y - bounding_box_size // 2))
        bottom_right = (min(image.shape[1], x + bounding_box_size // 2), min(image.shape[0], y + bounding_box_size // 2))
        cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(result_image, f"{i}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return result_image

def process_and_filter_images(image_pairs):
    processed_images = []
    for binarized_image, edge_image, candidates in image_pairs:
        visualize_mask_edges(binarized_image)
        filtered_candidates = filter_candidates(candidates, binarized_image, min_defect_size, max_defect_size)
        result_image = draw_bounding_boxes(edge_image, filtered_candidates)
        plt.imshow(result_image)
        plt.title("Filtered Defect Candidates")
        plt.axis('off')
        plt.show()
        processed_images.append((binarized_image, edge_image, filtered_candidates))
    return processed_images

filtered_ng_images_label1 = process_and_filter_images(labeled_ng_images_label1)
filtered_ng_images_label2 = process_and_filter_images(labeled_ng_images_label2)
filtered_ng_images_label3 = process_and_filter_images(labeled_ng_images_label3)
filtered_ok_images = process_and_filter_images(labeled_ok_images)
```

これらの関数は、マスクのエッジを可視化し、サイズとエッジの位置に基づいて欠陥候補をフィルタリングします。また、フィルタリングされた欠陥候補にバウンディングボックスを描画し、可視化します。

# 9. 欠陥候補の画像の保存

```python
def crop_and_enlarge_defect(image, candidate):
    y, x = map(int, candidate['centroid'])
    top_left = (max(0, x - crop_size // 2), max(0, y - crop_size // 2))
    bottom_right = (min(image.shape[1], x + crop_size // 2), min(image.shape[0], y + crop_size // 2))
    cropped = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    enlarged = cv2.resize(cropped, (crop_size * enlarge_factor, crop_size * enlarge_factor), interpolation=cv2.INTER_LINEAR)
    return enlarged

def save_defect_candidates(image_pairs, output_dir, image_label):
    os.makedirs(output_dir, exist_ok=True)
    all_features = []

    for i, (binarized_image, edge_image, candidates) in enumerate(image_pairs):
        image_name = f"image_{i}.jpg"
        for j, candidate in enumerate(candidates):
            enlarged_defect = crop_and_enlarge_defect(edge_image, candidate)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_defect_{j}.png"), enlarged_defect)

            features = {k: float(v) for k, v in candidate.items() if k != 'centroid' and k != 'bbox'}
            features['image_name'] = image_name
            features['Image_label'] = image_label
            features['defect_label'] = 0  # すべて0（OK）とする
            all_features.append(features)

    df = pd.DataFrame(all_features)
    df.to_csv(os.path.join(output_dir, 'defect_features.csv'), index=False)

save_defect_candidates(filtered_ng_images_label1, os.path.join(output_data_dir, 'ng_label1'), 1)
save_defect_candidates(filtered_ng_images_label2, os.path.join(output_data_dir, 'ng_label2'), 1)
save_defect_candidates(filtered_ng_images_label3, os.path.join(output_data_dir, 'ng_label3'), 1)
save_defect_candidates(filtered_ok_images, os.path.join(output_data_dir, 'ok'), 0)
```

これらの関数は、欠陥候補の画像を切り出し、拡大して保存します。また、欠陥候補の特徴量をCSVファイルとして保存します。

1. `crop_and_enlarge_defect`関数: 欠陥候補を中心に100x100ピクセルの正方形を切り出し、10倍に拡大します。
2. `save_defect_candidates`関数: 
   - 各欠陥候補を切り出し、拡大して保存します。
   - 欠陥候補の特徴量をディクショナリに格納します。
   - 全ての特徴量をDataFrameに変換し、CSVファイルとして保存します。

CSVファイルには以下の列が含まれます：
- image_name: 読み込んだ画像名
- 欠陥候補の特徴量（area, eccentricity, equivalent_diameter, euler_number, extent, filled_area, major_axis_length, minor_axis_length, orientation, perimeter, solidity）
- Image_label: 画像自体のOK/NGラベル（OK画像の場合0、NG画像の場合1）
- defect_label: 欠陥自体のOK/NGラベル（全て0（OK）とする）

このコードにより、鋳造部品の欠陥検出システムの前処理部分が実装され、欠陥候補の抽出、フィルタリング、保存までの一連の流れが自動化されます。得られた結果は、後続の機械学習モデルのトレーニングや詳細な分析に使用することができます。

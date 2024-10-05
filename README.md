了解しました。マスクのエッジ部分を欠陥候補から除外するように改良します。以下に全体のコードを示し、変更した部分を明示します。

# 1. ライブラリのインポート

```python
import os
import cv2
import numpy as np
from skimage import io, filters, feature, measure
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import pandas as pd
```

# 2. パラメータの設定

```python
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
gaussian_kernel_size = (7, 7)  # ガウシアンブラーのカーネルサイズ
canny_min_threshold = 30  # エッジ検出の最小しきい値
canny_max_threshold = 120  # エッジ検出の最大しきい値
sigma = 3  # ガウシアンブラーの標準偏差

# 欠陥サイズパラメータ
min_defect_size = 5  # 最小欠陥サイズ（0.5mm = 5px）
max_defect_size = 100  # 最大欠陥サイズ（10mm = 100px）

# テクスチャ検出パラメータ
texture_threshold = 15  # テクスチャの変化を検出するためのしきい値

# エッジ補完のパラメータ
edge_kernel_size = (3, 3)  # エッジ補完のカーネルサイズ
edge_open_iterations = 2   # ノイズ削除の繰り返し回数
edge_close_iterations = 2  # エッジ補完の繰り返し回数

# マスクエッジ検出のパラメータ
mask_edge_min_threshold = 100
mask_edge_max_threshold = 200
mask_edge_margin = 5  # マスクエッジの余裕幅（ピクセル単位）

# 欠陥候補の保存パラメータ
enlargement_factor = 10  # 欠陥候補画像の拡大倍率
```

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

# NG画像とOK画像をそれぞれ読み込む
ng_images_label1 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label1"))
ng_images_label2 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label2"))
ng_images_label3 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label3"))
ok_images = load_origin_keyence_images(os.path.join(input_data_dir, "OK"))
```

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
        cropped_image = image[:, :-crop_width]
        cropped_keyence_image = keyence_image[:, :-crop_width]
    else:
        cropped_image = image[:, crop_width:]
        cropped_keyence_image = keyence_image[:, crop_width:]
    
    return cropped_image, cropped_keyence_image

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

# NGとOK画像に対して二値化を実行
binarized_ng_images_label1 = binarize_images(updated_ng_images_label1)
binarized_ng_images_label2 = binarize_images(updated_ng_images_label2)
binarized_ng_images_label3 = binarize_images(updated_ng_images_label3)
binarized_ok_images = binarize_images(updated_ok_images)
```

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

def process_images_for_edge_detection(binarized_images):
    edged_images = []
    for binarized_image, cropped_keyence_image in binarized_images:
        edge_image = detect_edges_and_texture(cropped_keyence_image, binarized_image)
        edged_images.append((binarized_image, edge_image))
    return edged_images

# NGとOK画像に対してエッジ検出を実行
edged_ng_images_label1 = process_images_for_edge_detection(binarized_ng_images_label1)
edged_ng_images_label2 = process_images_for_edge_detection(binarized_ng_images_label2)
edged_ng_images_label3 = process_images_for_edge_detection(binarized_ng_images_label3)
edged_ok_images = process_images_for_edge_detection(binarized_ok_images)
```

# 7. エッジの補完とラベリング処理

```python
def create_mask_edge_margin(mask, margin):
    mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
    dilated_edges = cv2.dilate(mask_edges, kernel, iterations=1)
    return dilated_edges

def complete_edges(edge_image, mask):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    skeleton = skeletonize(edge_image > 0)
    kernel = np.ones(edge_kernel_size, np.uint8)
    
    # オープン処理でノイズを削除
    opened_skeleton = cv2.morphologyEx(skeleton.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=edge_open_iterations)
    
    # クローズ処理でエッジを接続
    connected_skeleton = cv2.morphologyEx(opened_skeleton, cv2.MORPH_CLOSE, kernel, iterations=edge_close_iterations)
    
    completed_edges = np.maximum(edge_image, connected_skeleton * 255)
    completed_edges = np.where(mask_edges_with_margin > 0, edge_image, completed_edges)
    return completed_edges.astype(np.uint8)

def label_and_measure_defects(edge_image, mask):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    binary_edge_image = (edge_image > 0).astype(np.uint8)
    binary_edge_image[mask_edges_with_margin > 0] = 0  # マスクエッジ部分を除外
    labels = measure.label(binary_edge_image, connectivity=2)
    defects = []
    for region in measure.regionprops(labels):
        y, x = region.bbox[0], region.bbox[1]
        h, w = region.bbox[2] - y, region.bbox[3] - x
        defect_info = {
            'label': region.label,
            'x': x, 'y': y, 'width': w, 'height': h,
            'area': region.area,
            'centroid_y': region.centroid[0], 'centroid_x': region.centroid[1],
            'perimeter': region.perimeter,
            'eccentricity': region.eccentricity,
            'orientation': region.orientation,
            'major_axis_length': region.major_axis_length,
            'minor_axis_length': region.minor_axis_length,
            'solidity': region.solidity,
            'extent': region.extent,
            'aspect_ratio': max(w, h) / min(w, h) if min(w, h) > 0 else 0,
            'max_length': max(w, h)
        }
        defects.append(defect_info)
    return defects

def process_images_for_labeling(edged_images):
    labeled_images = []
    for binarized_image, edge_image in edged_images:
        completed_edges = complete_edges(edge_image, binarized_image)
        defects = label_and_measure_defects(completed_edges, binarized_image)
        labeled_images.append((binarized_image, completed_edges, defects))
    return labeled_images

# NGとOK画像に対してラベリング処理を実行
labeled_ng_images_label1 = process_images_for_labeling(edged_ng_images_label1)
labeled_ng_images_label2 = process_images_for_labeling(edged_ng_images_label2)
labeled_ng_images_label3 = process_images_for_labeling(edged_ng_images_label3)
labeled_ok_images = process_images_for_labeling(edged_ok_images)
```

# 8. 欠陥候補のフィルタリング

```python
# 変更されたセクション：マスクエッジ部分の欠陥候補を除外する処理は不要になりました
def filter_defects_by_max_length(defects, min_size, max_size):
    return [defect for defect in defects if min_size <= defect['max_length'] <= max_size]

def process_images_for_filtering(labeled_images, image_type):
    filtered_images = []
    for i, (binarized_image, edge_image, defects) in enumerate(labeled_images):
        filtered_defects = filter_defects_by_max_length(defects, min_defect_size, max_defect_size)
        
        for j, defect in enumerate(filtered_defects, 1):
            defect['label'] = j
        
        image_name = f"{image_type}_{i}"
        filtered_images.append((image_name, binarized_image, edge_image, filtered_defects))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images_label1 = process_images_for_filtering(labeled_ng_images_label1, "ng_label1")
filtered_ng_images_label2 = process_images_for_filtering(labeled_ng_images_label2, "ng_label2")
filtered_ng_images_label3 = process_images_for_filtering(labeled_ng_images_label3, "ng_label3")
filtered_ok_images = process_images_for_filtering(labeled_ok_images, "ok")

# フィルタリング結果の可視化
def visualize_filtered_defects(image_name, image, defects, mask):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image, cmap='gray')
    
    # マスクのエッジを可視化
    mask_edges = create_mask_edge_margin(mask, mask_edge_margin)
    ax.imshow(mask_edges, alpha=0.3, cmap='cool')
    
    for defect in defects:
        rect = plt.Rectangle((defect['x'], defect['y']), defect['width'], defect['height'],
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(defect['x'], defect['y'], str(defect['label']), color='red', fontsize=12)
    
    plt.title(f"Filtered Defects with Mask Edges - {image_name}", fontsize=20)
    plt.axis('off')
    plt.show()

# フィルタリング結果の可視化（例：最初のNG画像）
if filtered_ng_images_label1:
    image_name, binarized_image, edge_image, filtered_defects = filtered_ng_images_label1[0]
    visualize_filtered_defects(image_name, edge_image, filtered_defects, binarized_image)
```

# 9. 欠陥候補の画像の保存とCSV出力

```python
def save_defect_image(image, defect, output_dir, image_name, defect_number):
    cx, cy = defect['centroid_x'], defect['centroid_y']
    size = max(defect['width'], defect['height'])
    
    x1 = max(int(cx - size), 0)
    y1 = max(int(cy - size), 0)
    x2 = min(int(cx + size), image.shape[1])
    y2 = min(int(cy + size), image.shape[0])
    
    defect_image = image[y1:y2, x1:x2]
    enlarged_image = cv2.resize(defect_image, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    
    output_filename = f"defect_{defect_number}.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, enlarged_image)
    
    return output_filename

def process_images_for_saving(filtered_images, base_output_dir, image_label):
    all_defects_data = []
    
    for image_name, binarized_image, edge_image, defects in filtered_images:
        image_type = image_name.split('_')[0]
        output_dir = os.path.join(base_output_dir, image_type, image_name)
        os.makedirs(output_dir, exist_ok=True)
        
        for defect in defects:
            output_filename = save_defect_image(edge_image, defect, output_dir, image_name, defect['label'])
            
            defect_data = {
                'image_name': image_name,
                'defect_image': os.path.join(image_type, image_name, output_filename),
                'Image_label': image_label,
                'defect_label': 0,  # デフォルトで0（OK）とする
            }
            defect_data.update(defect)
            all_defects_data.append(defect_data)
    
    return all_defects_data

# NGとOK画像の欠陥候補を保存し、データを収集
output_dir = os.path.join(output_data_dir, "defect_images")
os.makedirs(output_dir, exist_ok=True)

all_defects_data = []
all_defects_data.extend(process_images_for_saving(filtered_ng_images_label1, output_dir, 1))
all_defects_data.extend(process_images_for_saving(filtered_ng_images_label2, output_dir, 1))
all_defects_data.extend(process_images_for_saving(filtered_ng_images_label3, output_dir, 1))
all_defects_data.extend(process_images_for_saving(filtered_ok_images, output_dir, 0))

# CSVファイルに出力
df = pd.DataFrame(all_defects_data)
csv_output_path = os.path.join(output_data_dir, "defects_data.csv")
df.to_csv(csv_output_path, index=False)

print(f"Defects data saved to {csv_output_path}")
```


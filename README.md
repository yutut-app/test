以下は、項目6.2「ラベリング処理」と項目6.3「欠陥候補の中心座標の取得」を追加したコードです。全ての`キーエンス前処理画像(edges)`に対してラベリング処理を行い、サイズフィルタリングによってφ0.5mm以下、φ10mm以上の欠陥を除外し、更新後の画像ペアを表示します。

### .ipynb構成

#### 1. ライブラリのインポート
```python
# 必要なライブラリのインポート
%pip install -r requirements.txt

import os
import cv2
import numpy as np
from skimage import io, measure, feature
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
min_defect_size_mm = 0.5  # 最小の欠陥サイズ(mm)
max_defect_size_mm = 10.0  # 最大の欠陥サイズ(mm)
pixel_to_mm_ratio = 0.1  # 1pxあたりのmm単位
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

#### 6.2 ラベリング処理と 6.3 欠陥候補の中心座標の取得
```python
# ラベリング処理を行い、欠陥候補の中心座標を取得する関数
def label_and_find_defects(edges, min_size_px, max_size_px):
    # ラベリング処理を実行
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)
    
    # サイズフィルタリング
    defects = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_size_px < area < max_size_px:
            # 欠陥候補の中心座標と外接矩形の情報を保存
            left, top, width, height = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            center_x, center_y = centroids[i]
            defects.append({
                'center': (int(center_x), int(center_y)),
                'bounding_box': (left, top, width, height)
            })
    
    return defects

# 外接矩形を赤枠で描画する関数
def draw_defects_on_image(image, defects):
    img_with_defects = image.copy()
    for defect in defects:
        left, top, width, height = defect['bounding_box']
        # 赤枠で外接矩形を描画
        cv2.rectangle(img_with_defects, (left, top), (left + width, top + height), (0, 0, 255), 2)
        # 欠陥の中心座標を描画
        center_x, center_y = defect['center']
        cv2.circle(img_with_defects, (center_x, center_y), 3, (0, 0, 255), -1)
    return img_with_defects

# 欠陥のサイズフィルタリング基準を設定
min_size_px = int((min_defect_size_mm / pixel_to_mm_ratio) ** 2)
max_size_px = int((max_defect_size_mm / pixel_to_mm_ratio) ** 2)

# 全てのエッジ画像に対してラベリング処理を実行し、新しいリストを作成
def label_defects_in_images(edged_images):
    labeled_images = []
    for binarized_image, edges in edged_images:
        defects = label_and_find_defects(edges, min_size_px, max_size_px)
        image_with_defects = draw_defects_on_image(edges, defects)
        labeled_images.append((binarized_image, image_with_defects))
    return labeled_images

# NGとOK画像に対してラベリング処理を実行
labeled_ng_images_label1 = label_defects_in_images(edged_ng_images_label1)
labeled_ng_images_label2 = label_defects_in_images(edged_ng_images_label2)
labeled_ng_images_label3 = label_defects_in_images(edged_ng_images_label3)
labeled_ok_images = label_defects_in_images(edged_ok_images)
```

#### 7. 更新した画像ペアの表示
```python
# 更新されたNG_label1の最初の画像ペアを表示
if labeled_ng_images_label1:
    binarized_image, image_with_defects = labeled_ng_images_label1[0]
    
    # 二直化後の画像の表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(binarized_image, cmap='gray')
    plt.title("Binarized Image")
    plt.axis('off')

    # ラベリングと外接矩形を表示した画像の表示
    plt.subplot(1, 2, 2)
    plt.imshow(image_with_defects, cmap='gray')
    plt.title("Labeled Defects with Bounding Boxes")
    plt.axis('off')

    plt.show()
else:
    print("No images found after labeling.")
```

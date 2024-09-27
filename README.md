以下は、項目6.1「ラベリング処理」と6.2「欠陥候補の中心座標の取得」を追加し、全てのキーエンス前処理画像（cropped_keyence_image）に対してラベリング処理を行い、`binarized_images`を更新するコードです。その後、更新された最初のペアの画像を表示します。

### .ipynb構成

#### 1. ライブラリのインポート
```python
# 必要なライブラリのインポート
%pip install -r requirements.txt

import os
import cv2
import numpy as np
from skimage import io, measure, filters
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
min_defect_size_mm = 0.5  # 欠陥サイズの最小値（mm）
max_defect_size_mm = 10  # 欠陥サイズの最大値（mm）
pixel_to_mm = 0.1  # 1pxあたりのmm
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

H面（白い領域）のみを対象に欠陥候補の検出を行うため、`binarized_image`での白い部分（H面）を使い、`cropped_keyence_image`と照らし合わせて処理を進めます。このために、以下の流れで処理を進めます。

### 修正の要点
1. **H面の検出**：`binarized_image`の白い領域をH面とし、この領域内でのみ欠陥候補を検出します。
2. **キーエンス画像との対応**：`binarized_image`でH面が検出された部分にのみ、キーエンス画像上で欠陥候補をラベリングします。

### 修正後のコード

#### 6.1 ラベリング処理とH面の領域限定
```python
# H面の領域（白部分）を取得
def get_h_surface_region(binarized_image):
    # 白い部分（H面）を検出（ピクセル値が255の部分がH面）
    h_surface_mask = binarized_image == 255
    return h_surface_mask

# ラベリング処理（H面の領域に限定）
def label_defects_in_h_surface(binarized_image):
    # H面の領域（白部分）を取得
    h_surface_mask = get_h_surface_region(binarized_image)

    # H面の領域内のみラベリングを行う
    labeled_image, num_labels = measure.label(h_surface_mask, background=0, return_num=True)
    
    # 各欠陥領域のプロパティを取得
    properties = measure.regionprops(labeled_image)
    
    return labeled_image, properties
```

#### 6.2 欠陥候補の中心座標の取得と赤枠描画（H面に限定）
```python
# 欠陥のバウンディングボックス（赤枠）を描画（H面領域に限定）
def draw_defect_boxes_in_h_surface(image, defects):
    for defect in defects:
        minr, minc, maxr, maxc = defect.bbox  # ラベリングされた領域のバウンディングボックス
        image = cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 0, 255), 2)  # 赤枠で囲む
    return image

# 全てのcropped_keyence_imageに対してH面限定のラベリングと欠陥候補の描画を実行
def process_labeled_images_in_h_surface(binarized_image_pairs):
    processed_images = []
    for binarized_image, cropped_keyence_image in binarized_image_pairs:
        # binarized_imageに対してH面のみラベリング処理を行う
        labeled_image, properties = label_defects_in_h_surface(binarized_image)

        # サイズフィルタを使用して有効な欠陥を取得
        valid_defects = filter_defects_by_size(properties)

        # H面に基づき、欠陥のバウンディングボックス（赤枠）を描画
        labeled_keyence_image = draw_defect_boxes_in_h_surface(cropped_keyence_image, valid_defects)

        # 処理後の画像ペアを保存
        processed_images.append((labeled_image, labeled_keyence_image))

    return processed_images

# NGとOKの全てのペアに対してラベリング処理を実行
labeled_ng_images_label1 = process_labeled_images_in_h_surface(binarized_ng_images_label1)
#labeled_ng_images_label2 = process_labeled_images_in_h_surface(binarized_ng_images_label2)
#labeled_ng_images_label3 = process_labeled_images_in_h_surface(binarized_ng_images_label3)
#labeled_ok_images = process_labeled_images_in_h_surface(binarized_ok_images)
```

#### 7. 更新後の画像ペアの表示
```python
# 更新されたNG_label1の最初のペアを表示
if labeled_ng_images_label1:
    labeled_image, labeled_keyence_image = labeled_ng_images_label1[0]

    # ラベリング後のbinarized_imageの表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(labeled_image, cmap='gray')
    plt.title("Labeled Binarized Image (H Surface Only)")
    plt.axis('off')

    # ラベリング後のキーエンス画像の表示（赤枠で欠陥を表示）
    plt.subplot(1, 2, 2)
    plt.imshow(labeled_keyence_image, cmap='gray')
    plt.title("Labeled Keyence Image with Defect Boxes (H Surface Only)")
    plt.axis('off')

    plt.show()
else:
    print("No labeled images found.")
```

### 説明

1. **H面の領域限定ラベリング**：`get_h_surface_region`関数を使って、`binarized_image`内の白い領域（H面）を特定します。その領域に対してのみラベリング処理を行い、欠陥候補を検出します。
2. **赤枠描画**：`draw_defect_boxes_in_h_surface`関数で、H面内に存在する有効な欠陥のみを赤枠で囲んで描画します。
3. **ラベリング処理の流れ**：H面のみに対して欠陥の検出を行い、キーエンス前処理画像に欠陥を反映します。
4. **結果表示**：最初のペアのラベリング後の画像と、赤枠で囲まれたキーエンス画像を表示します。

これにより、H面に限定して欠陥候補を検出し、適切に赤枠で囲む処理が実行されるようになります。


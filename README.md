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
min_defect_size = 5  # 欠陥の最小サイズ（0.5mm）
max_defect_size = 100  # 欠陥の最大サイズ（10mm）
px_to_mm_ratio = 0.1  # 1ピクセル = 0.1mm
```

続きのコードを以下に示します。

#### 6.1 エッジ検出（続き）
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


#### 6.2 ラベリング処理
```python
# ラベリング処理を行い、欠陥候補を抽出する関数
def label_defects(edges):
    # ラベリング処理
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)
    
    # 有効な欠陥候補をリストに格納
    valid_defects = []
    for i in range(1, num_labels):  # ラベル0は背景なので無視
        # ラベルの領域サイズをpx_to_mm_ratioに基づいてmmに変換
        defect_area = stats[i, cv2.CC_STAT_AREA] * (px_to_mm_ratio ** 2)
        
        # 欠陥の面積が指定の範囲内にあるか確認
        if min_defect_size <= defect_area <= max_defect_size:
            valid_defects.append((stats[i], centroids[i]))
    
    return valid_defects

# 全てのエッジ画像に対してラベリング処理を実行し、新しいリストを作成
def label_defects_in_images(edged_images):
    labeled_images = []
    for binarized_image, edges in edged_images:
        defects = label_defects(edges)
        labeled_images.append((binarized_image, edges, defects))
    return labeled_images

# NGとOK画像に対してラベリング処理を実行
labeled_ng_images_label1 = label_defects_in_images(edged_ng_images_label1)
labeled_ng_images_label2 = label_defects_in_images(edged_ng_images_label2)
labeled_ng_images_label3 = label_defects_in_images(edged_ng_images_label3)
labeled_ok_images = label_defects_in_images(edged_ok_images)
```

#### 6.3 欠陥候補の中心座標の取得と赤枠表示
```python
# 欠陥候補を赤枠で表示する関数
def draw_defects_on_image(original_image, defects):
    # 画像をカラーに変換
    color_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    for defect in defects:
        stat, centroid = defect
        # 外接矩形を取得
        x, y, w, h = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP], stat[cv2.CC_STAT_WIDTH], stat[cv2.CC_STAT_HEIGHT]
        # 赤枠で矩形を描画
        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 中心座標を赤い点で表示
        cv2.circle(color_image, (int(centroid[0]), int(centroid[1])), 3, (0, 0, 255), -1)
    
    return color_image

# 最初のペアの画像に対して欠陥候補を表示
if labeled_ng_images_label1:
    binarized_image, edges, defects = labeled_ng_images_label1[0]
    
    # 元画像とエッジ検出結果に欠陥候補を表示
    defect_image = draw_defects_on_image(binarized_image, defects)
    
    # 画像の表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(binarized_image, cmap='gray')
    plt.title("Binarized Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(defect_image)
    plt.title("Defect Image with Red Rectangles")
    plt.axis('off')
    plt.show()
else:
    print("No images found with labeled defects.")
```

### 説明
1. **ラベリング処理**: `connectedComponentsWithStats`を使用して、エッジ検出された領域をラベリングし、各欠陥候補の領域と中心座標を取得します。欠陥サイズが指定された範囲（φ0.5mm以下、φ10mm以上は除外）にあるもののみを有効な欠陥候補として抽出します。
2. **赤枠での表示**: 欠陥候補に対して赤枠の外接矩形を描画し、さらに中心座標を赤い点で示します。`draw_defects_on_image`関数でこれを実行し、元画像に対して赤枠で欠陥候補が視覚的に確認できるようにしています。
3. **欠陥の除外基準**: 欠陥の面積は、ピクセル単位で計算されたものを`px_to_mm_ratio`でmmに変換し、指定された範囲に基づいて除外します。

これにより、エッジ検出後にラベリング処理を行い、欠陥候補を視覚的に表示することができます。

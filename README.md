欠陥がH面と高さが変わらず、小さいためエッジ検出で検出されない場合、エッジ検出に依存しない他の処理を追加して、欠陥をより正確に検出する必要があります。例えば、テクスチャの変化や輝度の変化を利用する方法があります。また、ガウシアンブラーやCannyエッジ検出のパラメータを調整して、より小さな欠陥も検出できるようにすることが考えられます。

以下に、パラメータの調整とテクスチャや輝度変化による追加処理を組み込んだコードを示します。

### 改良後の.ipynb構成

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
canny_min_threshold = 30  # エッジ検出の最小しきい値 (低めに調整)
canny_max_threshold = 120  # エッジ検出の最大しきい値 (低めに調整)
sigma = 3  # ガウシアンブラーの標準偏差 (低めに調整)
min_defect_size = 5  # 最小欠陥サイズ（0.5mm = 5px）
max_defect_size = 100  # 最大欠陥サイズ（10mm = 100px）
texture_threshold = 15  # テクスチャの変化を検出するためのしきい値
```

#### 6.1 エッジ検出とテクスチャ検出の改良
エッジ検出で欠陥が検出されないケースを補完するために、画像の輝度やテクスチャの変化を利用して欠陥を検出します。例えば、局所的な輝度差やラプラシアンフィルタによるエッジ強調を行い、小さい欠陥も見つけられるようにします。

```python
# エッジ検出に加えて、テクスチャの変化による欠陥検出を追加
def detect_edges_and_texture(cropped_keyence_image, binarized_image):
    # H面のマスクを適用して背景を除去
    masked_image = cv2.bitwise_and(cropped_keyence_image, cropped_keyence_image, mask=binarized_image)
    
    # ガウシアンブラーを適用してノイズを除去
    blurred_image = cv2.GaussianBlur(masked_image, gaussian_kernel_size, sigma)
    
    # エッジ検出（パラメータを低めに設定）
    edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
    
    # テクスチャの変化を検出（ラプラシアンを使用）
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    laplacian_edges = np.uint8(abs_laplacian > texture_threshold) * 255
    
    # エッジ検出とテクスチャ変化の結果を統合
    combined_edges = cv2.bitwise_or(edges, laplacian_edges)
    
    return combined_edges
```

#### 6.2 ラベリング処理と6.3 欠陥候補の中心座標の取得
```python
# 改良されたラベリング処理
def label_defects(edge_image, binarized_image, min_size, max_size):
    # ラベリング処理
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge_image)
    
    # マスクのエッジを除外
    labels_to_exclude = remove_mask_edges(labels, binarized_image)
    
    # サイズフィルタリング：欠陥が最小サイズと最大サイズの範囲内にあるか確認
    defect_candidates = []
    for i in range(1, num_labels):
        size = stats[i, cv2.CC_STAT_AREA]
        if min_size <= size <= max_size and i not in labels_to_exclude:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            defect_candidates.append((x, y, w, h, cx, cy))
    
    return defect_candidates

# 全てのedge_imageに対してラベリング処理を実行し、新しいリストを作成
def label_defects_in_images(edged_images):
    labeled_images = []
    for binarized_image, edge_image in edged_images:
        defects = label_defects(edge_image, binarized_image, min_defect_size, max_defect_size)
        labeled_images.append((binarized_image, edge_image, defects))
    return labeled_images
```

#### 7. 欠陥候補を赤枠で表示（改良後）
```python
# 欠陥候補を赤枠で表示する関数
def draw_defects(image, defects):
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # グレースケールをRGBに変換
    for (x, y, w, h, cx, cy) in defects:
        # 外接矩形を描画（赤色）
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 中心座標を描画
        cv2.circle(result_image, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    return result_image

# 更新されたNG_label1の最初の画像ペアに欠陥候補を表示
if labeled_ng_images_label1:
    binarized_image, edge_image, defects = labeled_ng_images_label1[0]
    
    # 欠陥候補を赤枠で描画
    result_image = draw_defects(edge_image, defects)
    
    # 結果を表示
    plt.figure(figsize=(10, 5))
    plt.imshow(result_image)
    plt.title("Detected Defects with Red Rectangles (including texture detection)")
    plt.axis('off')
    plt.show()
else:
    print("No defects found in the images.")
```

### 説明
1. **エッジ検出とテクスチャ検出の改良**:
    - `detect_edges_and_texture`関数で、エッジ検出に加えてラプラシアンフィルタを用いたテクスチャの変化を検出します。これにより、高さの変化がない小さな欠陥でも検出できるようにしています。
    - ガウシアンブラーのカーネルサイズやしきい値を調整して、小さな欠陥を見逃さないようにしています。

2. **欠陥候補のラベリング処理**:
    - ラベリング処理の際、エッジやテクスチャの変化を利用して、H面の境界部分を除外しながら欠陥候補を抽出します。

3. **赤枠での表示**:
    - 検出された欠陥候補に対して、赤枠と中心点を描画し、画像に表示します。

これで、エッジ検出されない小さな欠陥も含めて正確に検出できるように調整されました。

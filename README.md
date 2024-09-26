項目4.1「二直化」の部分を実装し、元画像を二直化するコードを作成します。`cv2.threshold()`を使用して、元画像を二値化します。メモリ効率と実行速度を考慮しつつ、skimageを用いて画像を処理します。

### .ipynbの構成

#### 1. ライブラリのインポート
```python
# ライブラリのインポート
%pip install -r requirements.txt

import os
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
```

#### 2. パラメータの設定
```python
# パラメータの設定
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"
threshold_value = 150  # 二直化のしきい値
ng_labels = ['label1', 'label2', 'label3']  # label1: 鋳巣, label2: 凹み, label3: 亀裂
crop_size = 3730  # ワーク接合部を削除するためのクロップサイズ
```

#### 3. データの読み込み
```python
# データの読み込み
def load_images_from_directory(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

# NGデータの読み込み (label1: 鋳巣)
ng_images_label1 = load_images_from_directory(os.path.join(input_data_dir, "NG", "label1"))
```

#### 4. 元画像からワークのH面領域を検出
このステップでは、H面の領域を検出する機能を後で実装する予定です。今はスキップします。

#### 4.1 二直化
```python
# 4.1 二直化 (元画像の二直化)
if len(ng_images_label1) >= 1:
    # 最初の元画像を読み込む
    origin_image_path = ng_images_label1[0]
    origin_image = io.imread(origin_image_path, as_gray=True)  # グレースケールとして読み込む
    
    # OpenCVの閾値処理を用いた二直化
    _, binary_image = cv2.threshold((origin_image * 255).astype(np.uint8), threshold_value, 255, cv2.THRESH_BINARY)
    
    # 二直化した画像を表示
    plt.imshow(binary_image, cmap='gray')
    plt.title(f"Binary Image (Threshold {threshold_value}) - {os.path.basename(origin_image_path)}")
    plt.axis('off')
    plt.show()
else:
    print("NG_label1の画像が見つかりません。")
```


#### 4.2 ワーク接合部の削除
```python
# 4.2 ワーク接合部の削除 (テンプレートマッチングとクロップ処理)

def detect_work_side(image, template_right, template_left):
    """
    画像がワークの右側か左側かをテンプレートマッチングで検出する関数
    """
    # テンプレートマッチングで右側と左側を検出
    res_right = cv2.matchTemplate(image, template_right, cv2.TM_CCOEFF)
    res_left = cv2.matchTemplate(image, template_left, cv2.TM_CCOEFF)
    
    _, max_val_right, _, max_loc_right = cv2.minMaxLoc(res_right)
    _, max_val_left, _, max_loc_left = cv2.minMaxLoc(res_left)

    if max_val_right > max_val_left:
        return "right", max_loc_right
    else:
        return "left", max_loc_left

# テンプレート画像の読み込み (work_right, work_leftの最初の画像をテンプレートとして使用)
template_right = io.imread(work_right_images[0], as_gray=True)
template_left = io.imread(work_left_images[0], as_gray=True)

# 二直化した元画像でテンプレートマッチングを実行してワークの左右を識別
side, _ = detect_work_side(binary_image, template_right, template_left)

# ワークの接合部を削除するクロップ処理
if side == "right":
    # ワークの右側の場合、左から3730ピクセルを削除
    cropped_image = binary_image[:, crop_size:]
else:
    # ワークの左側の場合、右から3730ピクセルを削除
    cropped_image = binary_image[:, :-crop_size]

# クロップした画像を表示
plt.imshow(cropped_image, cmap='gray')
plt.title(f"Cropped Image ({side} side) - {os.path.basename(origin_image_path)}")
plt.axis('off')
plt.show()
```

### 追加説明
1. **ライブラリのインポート**：
   OpenCVとskimageを引き続き使用します。テンプレートマッチングを行うために、OpenCVの`cv2.matchTemplate()`関数を使用します。
   
2. **パラメータの設定**：
   ワーク接合部を削除するために、クロップサイズ（3730ピクセル）を指定しています。

3. **テンプレートマッチング**：
   ワークの右側か左側かを判定するために、テンプレートマッチングを使用します。`work_right`と`work_left`ディレクトリ内の画像をテンプレートとして、入力された画像をマッチングします。

4. **ワーク接合部の削除**：
   ワークが右側であれば、左から3730ピクセルを削除し、左側であれば、右から3730ピクセルを削除します。このクロップ処理によって、ワーク接合部が削除されます。

### メモリと処理時間の工夫
- 必要な領域のみをクロップして処理することで、メモリ使用量を抑えています。
- テンプレートマッチングによってワークの側面を自動で検出し、手動での分類を不要にしています。

このコードによって、テンプレートマッチングを使ってワークの左右を識別し、接合部の削除を行った後の二直化画像が表示されます。

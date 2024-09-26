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

### 追加説明
1. **ライブラリのインポート**：
   必要なライブラリにOpenCV、skimage、matplotlibを追加します。`cv2`はOpenCVの関数で、二値化処理に使用します。
   
2. **パラメータの設定**：
   二直化のために、しきい値150を設定しています。データのディレクトリパスやNGラベルも設定します。

3. **データの読み込み**：
   `load_images_from_directory`関数を使って、NGラベル1（鋳巣）の画像を読み込みます。

4. **二直化処理**：
   OpenCVの`cv2.threshold()`関数を使用して、元画像を二値化します。ここで、`as_gray=True`を指定してグレースケール画像として読み込んだ後、しきい値150で二値化しています。

5. **画像の表示**：
   二値化した画像を`matplotlib`を使用して表示します。`cmap='gray'`を指定してグレースケールのまま表示されるようにしています。

### メモリと処理時間の工夫
- 画像は`skimage.io.imread`を使用してグレースケールで読み込むことで、余計なカラー情報を省き、メモリ使用量を削減しています。
- `cv2.threshold()`は高速で効率的な二直化処理が可能です。
- 処理は1ワークずつ進めるように設計されています。

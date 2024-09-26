以下のように、指定された処理をステップごとに進める.ipynbファイルを生成します。今回は、項目1〜3を行い、最後にNG_label1の画像を表示させる部分を実装します。

### .ipynbの構成

#### 1. ライブラリのインポート
```python
# Import necessary libraries
%pip install -r requirements.txt

import os
import cv2
import numpy as np
from skimage import io, filters, feature, measure
import matplotlib.pyplot as plt
```

#### 2. パラメータの設定
```python
# Set parameters
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"
ng_labels = ['label1', 'label2', 'label3']  # label1: porosity, label2: dents, label3: cracks (亀裂)
work_right_dir = os.path.join(input_data_dir, "work_right")
work_left_dir = os.path.join(input_data_dir, "work_left")
```

#### 3. データの読み込み
```python
# Function to load image paths from the directory
def load_images_from_directory(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

# Load NG images from each label directory
ng_images_label1 = load_images_from_directory(os.path.join(input_data_dir, "NG", "label1"))
ng_images_label2 = load_images_from_directory(os.path.join(input_data_dir, "NG", "label2"))
ng_images_label3 = load_images_from_directory(os.path.join(input_data_dir, "NG", "label3"))

# Load OK images
ok_images = load_images_from_directory(os.path.join(input_data_dir, "OK"))
```

#### 4. NG_label1の画像表示
```python
# Display first NG image from label1 (porosity) with correct color
if ng_images_label1:
    image_path = ng_images_label1[0]
    image = io.imread(image_path)
    
    # Check if image is in the correct color format (RGB)
    if len(image.shape) == 3:  # This is for color images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    plt.title(f"NG Label 1 (Porosity) - {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()

```

### requirements.txt
```plaintext
opencv-python
scikit-image
matplotlib
numpy
```

### 説明
1. **ライブラリのインポート**では、`requirements.txt`から指定のライブラリをインポートし、`skimage`や`opencv`を使って画像処理を行います。
2. **パラメータの設定**では、画像のディレクトリやラベル情報、作業するワーク側のディレクトリパスを設定します。
3. **データの読み込み**では、NGとOK画像をそれぞれ読み込み、パスをリストに格納します。NGのラベル別に分かれた画像ファイルを読み込みます。
4. **NG_label1の画像表示**では、ラベル1のNG（鋳巣）の画像を読み込み、表示します。

この構成で、次のステップに進める基盤が整います。

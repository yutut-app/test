以下は、修正された.ipynbファイルで、元画像（Normal）とキーエンス前処理画像（Shape）が同じワークであることを認識し、それらを同時に処理するためのステップです。

### .ipynb構成

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
# Function to load matching Normal and Shape images based on filename pattern
def load_origin_keyence_images(directory):
    normal_images = {}
    shape_images = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "Normal" in file and file.endswith(".jpg"):
                # Extract the base name to match with the Shape file
                base_name = file.replace("Normal", "")
                normal_images[base_name] = os.path.join(root, file)
            elif "Shape" in file and file.endswith(".jpg"):
                base_name = file.replace("Shape", "")
                shape_images[base_name] = os.path.join(root, file)
    
    # Find matching Normal and Shape images
    matched_images = []
    for base_name in normal_images:
        if base_name in shape_images:
            matched_images.append((normal_images[base_name], shape_images[base_name]))
    return matched_images

# Load NG images from label1 (porosity)
ng_images_label1 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label1"))

# Load OK images
ok_images = load_origin_keyence_images(os.path.join(input_data_dir, "OK"))
```

#### 4. NG_label1の画像表示
```python
# Display first NG images from label1 (porosity) - both origin and Keyence processed images
if ng_images_label1:
    # Get the first pair of Normal and Shape images
    origin_image_path, keyence_image_path = ng_images_label1[0]
    
    # Load the images
    origin_image = io.imread(origin_image_path)
    keyence_image = io.imread(keyence_image_path)

    # Display original image (Normal)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(origin_image, cmap='gray')
    plt.title(f"Origin Image - {os.path.basename(origin_image_path)}")
    plt.axis('off')

    # Display Keyence processed image (Shape)
    plt.subplot(1, 2, 2)
    plt.imshow(keyence_image, cmap='gray')
    plt.title(f"Keyence Image - {os.path.basename(keyence_image_path)}")
    plt.axis('off')

    plt.show()
else:
    print("No images found in NG_label1.")
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
2. **パラメータの設定**では、画像のディレクトリやラベル情報、ワーク側のディレクトリパスを設定します。
3. **データの読み込み**では、`Normal`と`Shape`がペアになった画像をそれぞれ正しくマッチングして読み込みます。同じワークに対応する元画像（Normal）とキーエンス前処理画像（Shape）を基に、マッチングペアをリストに格納します。
4. **NG_label1の画像表示**では、ペアとなったNormal（元画像）とShape（キーエンス前処理画像）の最初のセットを表示します。

この構成で、元画像とキーエンス前処理画像が同じワークとして正しく処理されるようになり、次のステップに進める基盤が整います。

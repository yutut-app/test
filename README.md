--

### **スライド1：サマリ**
**話す内容**:
「今日は、進行中のプロジェクトの進捗についてお話しします。具体的には、目的や課題を整理し、どのようにモデルを開発していくかのフローをご紹介します。また、現在進めている欠陥検出の実装や、性能指標についても提案し、最後に今後の予定について説明します。これを通じて、現状の理解を深めてもらえるようにします。」

---

### **スライド2：事例紹介**
**話す内容**:
「こちらは、欠陥検出を行う際のオムロン事例を示したものです。ブラウザの方で紹介します。」

---

### **スライド3：提案するモデルのフロー**
**話す内容**:
「こちらが、今回提案するモデルのフローです。入力画像に対してまず欠陥候補を検出し、その後に画像を切り出して処理します。その切り出された画像をAIに入力し、ツールマークや鋳巣などを分類します。最終的には、NGとされるφ0.5mm以上の鋳巣を検出することが目的です。このフローに従って、欠陥検出を行う形になります。」

---

### **スライド4：性能指標の提案**
**話す内容**:
「次に、モデルの性能を評価するための指標を提案します。重要な指標は2つです。まず『見逃し率』は、NG対象を見逃してしまう確率です。これを0.00%にすることが目標です。次に『見過ぎ率』は、NG対象でないものを誤ってNGと判断してしまう確率です。これをできる限り小さく抑えたいと考えています。この2つの指標を使って、モデルの性能を測定します。」

---

### **スライド5：欠陥候補の検出フロー**
**話す内容**:
「こちらが、欠陥候補を検出するためのフローです。まずは、エッジ検出やしきい値処理といった画像処理を使って、ワーク表面に異常な部分がないかを調べます。しかし、ワークの凹凸や背景がある場合、単純なエッジ検出ではすべての欠陥を正確に検出できるわけではないため、次に別の手法や深層学習を活用するかもしれません。現在、どの手法が最適か検討中です。」

---

### **スライド6：画像の切り出しフロー**
**話す内容**:
「ここでは、欠陥候補が検出された後、その部分を固定サイズの正方形に切り出す流れを説明しています。固定サイズに統一することで、AIモデルに入力するデータを揃えることができます。ただし、画像が小さい場合はパディングを行い、逆に大きすぎる場合は縮小して調整します。このプロセスは、後の分類精度を上げるために重要です。」

---

### **スライド追加：LinuxとWindowsのメリット・デメリット**
**話す内容**:
「ここでは、LinuxとWindowsの両方でモデルを使用する際のメリットとデメリットを比較しています。LinuxはGPUやコンテナ技術の対応が強く、パフォーマンスに優れていますが、操作が難しいというデメリットがあります。一方、Windowsは使いやすく、多くのソフトウェアに対応していますが、パフォーマンスやセキュリティ面で劣る部分があります。特に現場で使いやすいのはWindowsかと思いますが、状況に応じて選ぶ必要があります。」

---

### **スライド7：今後の予定**
**話す内容**:
「最後に、今後の予定について説明します。現在、欠陥候補の検出や画像の切り出しを中心に実装を進めています。この部分の精度を向上させつつ、次のステップであるAIによる欠陥分類に移っていきます。また、現場での使用を考慮して、Windowsでの運用も進めていく予定です。」

---

以下は、指定された要件に基づいて、Pythonコードと `requirements.txt` の内容を含む実装です。処理は6つの項目に分けられ、処理時間とメモリ使用量を最小限に抑える工夫をしています。

### 1. requirements.txt

まずは必要なライブラリを `requirements.txt` に記載します。

```txt
opencv-python==4.5.3.56
numpy==1.21.2
matplotlib==3.4.3
scikit-image==0.18.3
```

### 2. Pythonコード (`.ipynb`)

次に、Jupyter NotebookにおけるPythonコードです。

```python
# 1. ライブラリのインポート
%pip install -r requirements.txt

import os
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

# 2. パラメータの設定
# 画像のスケール（例: 1px = 0.01mm）
PIXEL_TO_MM = 0.01
# 欠陥の最小径（0.5mm）
MIN_DEFECT_DIAMETER_MM = 0.5
MIN_DEFECT_AREA_PX = (MIN_DEFECT_DIAMETER_MM / PIXEL_TO_MM) ** 2 * np.pi  # 欠陥の最小面積(px²)

# データのパス
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"

# OKとNGのディレクトリ設定
ok_dir = os.path.join(input_data_dir, 'OK')
ng_dir = os.path.join(input_data_dir, 'NG')
output_ok_dir = os.path.join(output_data_dir, 'OK')
output_ng_dir = os.path.join(output_data_dir, 'NG')

# 3. データの読み込み
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 画像はグレースケールで読み込む
            images.append((filename, img))
    return images

# OK画像とNG画像の読み込み
ok_images = load_images_from_directory(ok_dir)

# NG画像のサブディレクトリ（鋳巣、凹み、亀裂）から読み込み
ng_subdirs = ['鋳巣', '凹み', '亀裂']
ng_images = {subdir: load_images_from_directory(os.path.join(ng_dir, subdir)) for subdir in ng_subdirs}

# 4. 欠陥候補の検出
def detect_defect_candidates(image):
    # エッジ検出を使用して欠陥候補を抽出
    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred_img, 50, 150)
    
    # ラベリングによる輪郭抽出
    labeled_image = measure.label(edges, connectivity=2)
    properties = measure.regionprops(labeled_image)
    
    # 欠陥候補の輪郭と面積を確認
    defect_candidates = []
    for prop in properties:
        if prop.area >= MIN_DEFECT_AREA_PX:
            defect_candidates.append(prop.bbox)  # bbox = (min_row, min_col, max_row, max_col)
    
    return defect_candidates

# 5. 画像切り出し
def crop_defect_region(image, bbox):
    min_row, min_col, max_row, max_col = bbox
    cropped_image = image[min_row:max_row, min_col:max_col]
    return cropped_image

# 6. 切り出した画像の保存
def save_cropped_image(image, filename, defect_type, is_ng=False):
    if is_ng:
        save_dir = os.path.join(output_ng_dir, defect_type)
    else:
        save_dir = output_ok_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, image)

# OK画像に対する処理
for filename, image in ok_images:
    defect_candidates = detect_defect_candidates(image)
    for i, bbox in enumerate(defect_candidates):
        cropped_image = crop_defect_region(image, bbox)
        save_cropped_image(cropped_image, f"{filename}_defect_{i}.jpg", "OK", is_ng=False)

# NG画像に対する処理
for defect_type, images in ng_images.items():
    for filename, image in images:
        defect_candidates = detect_defect_candidates(image)
        for i, bbox in enumerate(defect_candidates):
            cropped_image = crop_defect_region(image, bbox)
            save_cropped_image(cropped_image, f"{filename}_defect_{i}.jpg", defect_type, is_ng=True)

print("処理が完了しました。")
```

### 各項目の説明

1. **ライブラリのインポート**：
   `requirements.txt` から必要なライブラリをインストールし、Pythonコード内で使用するライブラリをインポートします。

2. **パラメータの設定**：
   - 画像スケールとして1pxあたりの物理的なサイズ（mm）を設定。
   - φ0.5mm以上の欠陥を検出するために必要なピクセル数を計算し、基準とします。

3. **データの読み込み**：
   - OKとNGのディレクトリから画像を読み込み、OKデータとNGデータをそれぞれ格納します。
   
4. **欠陥候補の検出**：
   - 画像処理を行い、エッジ検出を使って欠陥候補となる領域を抽出。
   - ラベリングを行い、各領域の面積を計算してφ0.5mm以上の欠陥を検出。

5. **画像切り出し**：
   - 検出された欠陥領域を基に、画像を切り出します。

6. **切り出した画像の保存**：
   - 切り出した画像をOKまたはNGの対応するディレクトリに保存します。NGの場合は欠陥の種類別にディレクトリを分けています。

このように、各項目が独立して動作しつつ、メモリや処理時間を効率的に使用できるよう工夫しています。

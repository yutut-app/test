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

以下の内容に基づいて、Pythonコードと`requirements.txt`ファイルを生成します。

### 1. ライブラリのインポート

```python
import os
import cv2
import numpy as np
from pathlib import Path
```

### 2. パラメータの設定

```python
# 入力画像のパス
input_path = Path(r"../data/input")
output_path = Path(r"../data/output")

# 出力ディレクトリを作成
output_ok_path = output_path / "OK"
output_ng_path = output_path / "NG"

output_ok_path.mkdir(parents=True, exist_ok=True)
output_ng_path.mkdir(parents=True, exist_ok=True)

# 欠陥サイズの閾値（0.5mm以上のものを検出）
min_defect_size = 0.5  # mm
pixel_to_mm_ratio = 0.01  # 仮の変換率、実際の値に応じて調整
```

### 3. データの読み込み

```python
def load_images_from_directory(directory):
    image_paths = list(directory.glob("*.jpg"))
    images = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append((img_path, img))
    return images

ok_images = load_images_from_directory(input_path / "OK")
ng_images = []
for defect_type in ["鋳巣", "凹み", "亀裂"]:
    defect_images = load_images_from_directory(input_path / "NG" / defect_type)
    ng_images.extend(defect_images)
```

### 4. 欠陥候補の検出

```python
def detect_defects(image):
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Cannyによるエッジ検出
    edges = cv2.Canny(gray, 100, 200)
    
    # 輪郭抽出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    defect_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 面積を使ってフィルタリング（実際の面積の基準は調整が必要）
        if area * pixel_to_mm_ratio > min_defect_size:
            defect_contours.append(cnt)
    return defect_contours
```

### 5. 画像切り出し

```python
def crop_and_save_image(image, contours, output_dir, image_name):
    for i, cnt in enumerate(contours):
        # 輪郭に基づき、画像を切り出す
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_image = image[y:y+h, x:x+w]
        
        # ファイル名を生成して保存
        cropped_image_path = output_dir / f"{image_name.stem}_defect_{i}.jpg"
        cv2.imwrite(str(cropped_image_path), cropped_image)
```

### 6. 切り出した画像の保存

```python
# OK画像の処理
for img_path, img in ok_images:
    contours = detect_defects(img)
    crop_and_save_image(img, contours, output_ok_path, img_path)

# NG画像の処理
for img_path, img in ng_images:
    contours = detect_defects(img)
    crop_and_save_image(img, contours, output_ng_path, img_path)
```

---

### `requirements.txt` ファイル
```txt
opencv-python==4.5.3.56
numpy==1.21.2
```

これで、コードは前処理済みの画像を読み込み、欠陥候補を検出し、画像を切り出して保存する流れを実装しています。

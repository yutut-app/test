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

ワーク全体ではなく、ワーク内の小さな欠陥を検出するためには、以下のように改良を加えます。

### 改良点

1. **背景とワーク部分を分離**：
   ワーク全体がエッジ検出によって検出されてしまう問題を解決するため、背景とワークを区別して、小さな欠陥だけを検出するようにします。これを実現するためには、まずワークの輪郭を検出し、その内側にある欠陥だけを検出するようにします。

2. **欠陥部分のフィルタリング**：
   ワーク全体の領域を無視し、小さな欠陥部分のみを対象とするために、欠陥の面積が小さい領域だけを検出対象にします。

3. **ワーク内部の欠陥検出**：
   ワーク全体の輪郭を利用して、その内側のみでエッジ検出を行うことで、ワーク内部の欠陥を見つけるようにします。

# 1. ライブラリのインポート
%pip install -r requirements.txt

import os
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import gc  # ガベージコレクションを利用するためのモジュール

# 2. パラメータの設定
# 1px = 0.0625mm
PIXEL_TO_MM = 0.0625
# 欠陥の最小径（0.5mm）と最大径（50mm）
MIN_DEFECT_DIAMETER_MM = 0.5
MAX_DEFECT_DIAMETER_MM = 50
MIN_DEFECT_AREA_PX = (MIN_DEFECT_DIAMETER_MM / PIXEL_TO_MM) ** 2 * np.pi  # 欠陥の最小面積(px²)
MAX_DEFECT_AREA_PX = (MAX_DEFECT_DIAMETER_MM / PIXEL_TO_MM) ** 2 * np.pi  # 欠陥の最大面積(px²)
BATCH_SIZE = 50  # メモリ節約のために一度に読み込む画像の数を制限

# データのパス
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"
test_data_dir = r"../data/output/test/work_frame"  # ワークの輪郭画像を保存するディレクトリ

# OKとNGのディレクトリ設定
ok_dir = os.path.join(input_data_dir, 'OK')
ng_dir = os.path.join(input_data_dir, 'NG')
output_ok_dir = os.path.join(output_data_dir, 'OK')
output_ng_dir = os.path.join(output_data_dir, 'NG')

# 必要なディレクトリを作成（存在しない場合）
os.makedirs(test_data_dir, exist_ok=True)

# 3. データの読み込み（メモリ効率化のためバッチ処理）
def load_images_from_directory(directory, batch_size=BATCH_SIZE, resize_factor=0.5):
    filenames = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    for i in range(0, len(filenames), batch_size):
        batch_filenames = filenames[i:i+batch_size]
        images = []
        for filename in batch_filenames:
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 画像はグレースケールで読み込む
            if img is None:
                continue
            # 画像のリサイズ（メモリ節約のために縮小）
            img_resized = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            images.append((filename, img_resized))
        yield images
        gc.collect()  # バッチ処理の後にメモリを解放

# OK画像とNG画像のバッチ読み込み
def process_ok_images():
    for batch_images in load_images_from_directory(ok_dir):
        process_images(batch_images, "OK", is_ng=False)

def process_ng_images():
    for subdir in ['鋳巣', '凹み', '亀裂']:
        subdir_path = os.path.join(ng_dir, subdir)
        for batch_images in load_images_from_directory(subdir_path):
            process_images(batch_images, subdir, is_ng=True)

# 4. 欠陥候補の検出（ワークの中の小さな欠陥）
def detect_defect_candidates(image, filename):
    # Step 1: ワークの輪郭を検出する（背景を除外）
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ワークの最大の輪郭（外枠）を取得
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # ワーク輪郭画像を保存
        work_contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 輪郭を色付き画像に変換
        cv2.drawContours(work_contour_image, [largest_contour], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(test_data_dir, f"{filename}_work_contour.jpg"), work_contour_image)
        
        # Step 2: ワーク内部でエッジ検出を行う
        work_area = cv2.bitwise_and(image, mask)
        blurred_img = cv2.GaussianBlur(work_area, (5, 5), 0)
        edges = cv2.Canny(blurred_img, 50, 150)
    
        # ラベリングによる輪郭抽出
        labeled_image = measure.label(edges, connectivity=2)
        properties = measure.regionprops(labeled_image)
        
        # 欠陥候補の輪郭と面積を確認
        defect_candidates = []
        for prop in properties:
            if MIN_DEFECT_AREA_PX <= prop.area <= MAX_DEFECT_AREA_PX:  # 欠陥が指定範囲内の大きさの場合
                defect_candidates.append(prop.bbox)  # bbox = (min_row, min_col, max_row, max_col)
        
        return defect_candidates
    else:
        return []  # 輪郭が見つからない場合は候補なし

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

# 画像を処理する関数
def process_images(batch_images, defect_type, is_ng=False):
    for filename, image in batch_images:
        defect_candidates = detect_defect_candidates(image, filename)
        for i, bbox in enumerate(defect_candidates):
            cropped_image = crop_defect_region(image, bbox)
            save_cropped_image(cropped_image, f"{filename}_defect_{i}.jpg", defect_type, is_ng)

# OK画像の処理
process_ok_images()

# NG画像の処理
process_ng_images()

print("処理が完了しました。")

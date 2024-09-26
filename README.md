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
input_data_dir = os.path.join("..", "data", "input")
output_data_dir = os.path.join("..", "data", "output")
test_data_dir = os.path.join(output_data_dir, "test", "work_frame")  # ワークの輪郭画像を保存するディレクトリ

# ワークのテンプレート画像
template_right_dir = os.path.join(input_data_dir, 'work_frame_right')  # 右側のテンプレート
template_left_dir = os.path.join(input_data_dir, 'work_frame_left')    # 左側のテンプレート

# OKとNGのディレクトリ設定
ok_dir = os.path.join(input_data_dir, 'OK')
ng_dir = os.path.join(input_data_dir, 'NG')
output_ok_dir = os.path.join(output_data_dir, 'OK')
output_ng_dir = os.path.join(output_data_dir, 'NG')

# 必要なディレクトリを作成（存在しない場合）
os.makedirs(test_data_dir, exist_ok=True)

# 3. テンプレートマッチングを行う関数
def template_matching(img, templates):
    best_match_val = -1
    best_match_loc = None
    best_template = None
    
    for template_path in templates:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Warning: Failed to load template image {template_path}")
            continue

        # テンプレートマッチングを実行
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > best_match_val:  # より良い一致を見つけた場合
            best_match_val = max_val
            best_match_loc = max_loc
            best_template = template

    return best_template, best_match_loc

# 4. データの読み込み（メモリ効率化のためバッチ処理）
def load_images_from_directory(directory, batch_size=BATCH_SIZE, resize_factor=0.5):
    filenames = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    print(f"Found {len(filenames)} images in {directory}")  # デバッグ用出力

    for i in range(0, len(filenames), batch_size):
        batch_filenames = filenames[i:i+batch_size]
        images = []
        for filename in batch_filenames:
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 画像はグレースケールで読み込む
            if img is None:
                print(f"Warning: Failed to load image {img_path}")  # 読み込みに失敗した場合
                continue
            # 画像のリサイズ（メモリ節約のために縮小）
            img_resized = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            images.append((filename, img_resized))
        print(f"Batch size: {len(images)} images processed")  # デバッグ用出力
        yield images
        gc.collect()  # バッチ処理の後にメモリを解放

# OK画像とNG画像のバッチ読み込み
def process_ok_images():
    for batch_images in load_images_from_directory(ok_dir):
        process_images(batch_images, "OK", is_ng=False)

def process_ng_images():
    for subdir in ['porosity', 'dent', 'crack']:  # 英語名に変更
        subdir_path = os.path.join(ng_dir, subdir)
        print(f"Processing NG images in {subdir_path}")
        for batch_images in load_images_from_directory(subdir_path):
            process_images(batch_images, subdir, is_ng=True)

# 5. ワーク検出のためのテンプレートマッチング（右側と左側のワークをチェック）
def detect_workpiece(image, filename):
    # テンプレート画像を取得
    templates = [os.path.join(template_right_dir, f) for f in os.listdir(template_right_dir) if f.endswith('.jpg')] + \
                [os.path.join(template_left_dir, f) for f in os.listdir(template_left_dir) if f.endswith('.jpg')]
    
    best_template, best_loc = template_matching(image, templates)

    if best_template is not None:
        # テンプレートマッチング結果を描画
        h, w = best_template.shape[:2]
        cv2.rectangle(image, best_loc, (best_loc[0] + w, best_loc[1] + h), (0, 0, 255), 3)

        # 検出結果を保存
        cv2.imwrite(os.path.join(test_data_dir, f"{filename}_work_detected.jpg"), image)
        return True
    else:
        print(f"No matching template found for {filename}")
        return False

# 6. 欠陥候補の検出（ワークの中の小さな欠陥）
def detect_defect_candidates(image, filename):
    if not detect_workpiece(image, filename):
        return []  # ワークが検出できなかった場合は欠陥なし

    # ワーク検出後に欠陥検出を行う（検出したワーク領域内でエッジ検出を行う）
    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
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

# 7. 画像切り出し
def crop_defect_region(image, bbox):
    min_row, min_col, max_row, max_col = bbox
    cropped_image = image[min_row:max_row, min_col:max_col]
    return cropped_image

# 8. 切り出した画像の保存
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


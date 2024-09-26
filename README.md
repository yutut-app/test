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

# 3. エッジ検出を行う関数（テンプレート画像に対しても適用）
def apply_edge_detection(image):
    blurred_img = cv2.GaussianBlur(image, (15, 15), 0)
    edges = cv2.Canny(blurred_img, 15, 300)
    return edges

# 4. テンプレートマッチングを行う関数（エッジ検出したテンプレートを使用）
def template_matching(img, templates):
    best_match_val = -1
    best_match_loc = None
    best_template = None
    
    for template_path in templates:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Warning: Failed to load template image {template_path}")
            continue

        # テンプレート画像にエッジ検出を適用
        edge_template = apply_edge_detection(template)

        # テンプレートマッチングを実行
        res = cv2.matchTemplate(img, edge_template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > best_match_val:  # より良い一致を見つけた場合
            best_match_val = max_val
            best_match_loc = max_loc
            best_template = edge_template

    return best_template, best_match_loc

# 5. データの読み込み（メモリ効率化のためバッチ処理）
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

# 6. ワーク検出のためのテンプレートマッチング（エッジ検出したテンプレートを使用）
def detect_workpiece(image, filename):
    # 入力画像にもエッジ検出を適用
    edge_image = apply_edge_detection(image)

    # テンプレート画像を取得（エッジ検出したものを使用）
    templates = [os.path.join(template_right_dir, f) for f in os.listdir(template_right_dir) if f.endswith('.jpg')] + \
                [os.path.join(template_left_dir, f) for f in os.listdir(template_left_dir) if f.endswith('.jpg')]
    
    best_template, best_loc = template_matching(edge_image, templates)

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

# 7. 欠陥候補の検出（ワークの中の小さな欠陥）
def detect_defect_candidates(image, filename):
    if not detect_workpiece(image, filename):
        return []  # ワークが検出できなかった場合は欠陥なし

    # ワーク検出後に欠陥検出を行う（検出したワーク領域内でエッジ検出を行う）
    blurred_img = cv2.GaussianBlur(image, (15, 15), 0)
    edges = cv2.Canny(blurred_img, 15, 300)
    
    # ラベリングによる輪郭抽出
    labeled_image = measure.label(edges, connectivity=2)
    properties = measure.regionprops(labeled_image)
    
    # 欠陥候補の輪郭と面積を確認
    defect_candidates = []
    for prop in properties:
        if MIN_DEFECT_AREA_PX <= prop.area <= MAX_DEFECT_AREA_PX:  # 欠陥が指定範囲内の大きさの場合
            defect_candidates.append(prop.bbox)  # bbox = (min_row, min_col, max_row, max_col)

    return defect_candidates

# 8. 画像切り出し
def crop_defect_region(image, bbox):
    min_row, min_col, max_row, max_col = bbox
    cropped_image = image[min_row:max_row, min_col:max_col]
    return cropped_image

# 9. 切り出した画像の保存
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

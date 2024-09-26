# 1. ライブラリのインポート
%pip install -r requirements.txt

import os
import cv2
import numpy as np
from skimage import measure
import gc  # ガベージコレクションを利用するためのモジュール

# 2. パラメータの設定
PIXEL_TO_MM = 0.0625  # 1px = 0.0625mm
MIN_DEFECT_DIAMETER_MM = 0.5
MAX_DEFECT_DIAMETER_MM = 50
MIN_DEFECT_AREA_PX = (MIN_DEFECT_DIAMETER_MM / PIXEL_TO_MM) ** 2 * np.pi
MAX_DEFECT_AREA_PX = (MAX_DEFECT_DIAMETER_MM / PIXEL_TO_MM) ** 2 * np.pi
BATCH_SIZE = 50

# データのパス
input_data_dir = os.path.join("..", "data", "input")
output_data_dir = os.path.join("..", "data", "output")
test_data_dir = os.path.join(output_data_dir, "test", "work_frame")  # ワークの輪郭画像を保存するディレクトリ

# ワークのテンプレート画像（エッジ検出したものを使用）
template_right_dir = os.path.join(input_data_dir, 'work_frame_right')
template_left_dir = os.path.join(input_data_dir, 'work_frame_left')

# OKとNGのディレクトリ設定
ok_dir = os.path.join(input_data_dir, 'OK')
ng_dir = os.path.join(input_data_dir, 'NG')
output_ok_dir = os.path.join(output_data_dir, 'OK')
output_ng_dir = os.path.join(output_data_dir, 'NG')

# 必要なディレクトリを作成（存在しない場合）
os.makedirs(test_data_dir, exist_ok=True)

# 3. エッジ検出を行う関数（テンプレート画像にも適用）
def apply_edge_detection(image):
    blurred_img = cv2.GaussianBlur(image, (15, 15), 0)
    edges = cv2.Canny(blurred_img, 15, 300)
    return edges

# 4. テンプレートマッチングを行う関数
def template_matching(img, templates):
    best_match_val = -1
    best_match_loc = None
    best_template = None
    
    for template_path in templates:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Warning: Failed to load template image {template_path}")
            continue

        # エッジ検出をテンプレートに適用
        template_edges = apply_edge_detection(template)

        # テンプレートマッチングを実行
        res = cv2.matchTemplate(img, template_edges, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > best_match_val:  # より良い一致を見つけた場合
            best_match_val = max_val
            best_match_loc = max_loc
            best_template = template_edges

    return best_template, best_match_loc

# 5. ワーク検出のためのテンプレートマッチング（右側と左側のワークをチェック）
def detect_workpiece(image, filename):
    # テンプレート画像を取得
    templates = [os.path.join(template_right_dir, f) for f in os.listdir(template_right_dir) if f.endswith('.jpg')] + \
                [os.path.join(template_left_dir, f) for f in os.listdir(template_left_dir) if f.endswith('.jpg')]
    
    # 入力画像にエッジ検出を適用
    edge_image = apply_edge_detection(image)
    
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

# 6. 欠陥候補の検出
def detect_defect_candidates(image, filename):
    if not detect_workpiece(image, filename):
        return []  # ワークが検出できなかった場合は欠陥なし

    # エッジ検出を適用して欠陥検出
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

# 7. 画像切り出しと保存（省略可能な部分）
def crop_defect_region(image, bbox):
    min_row, min_col, max_row, max_col = bbox
    cropped_image = image[min_row:max_row, min_col:max_col]
    return cropped_image

def save_cropped_image(image, filename, defect_type, is_ng=False):
    if is_ng:
        save_dir = os.path.join(output_ng_dir, defect_type)
    else:
        save_dir = output_ok_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, image)

# 画像を処理する関数（省略可能な部分）
def process_images(batch_images, defect_type, is_ng=False):
    for filename, image in batch_images:
        defect_candidates = detect_defect_candidates(image, filename)
        for i, bbox in enumerate(defect_candidates):
            cropped_image = crop_defect_region(image, bbox)
            save_cropped_image(cropped_image, f"{filename}_defect_{i}.jpg", defect_type, is_ng)

# OK画像の処理（省略可能な部分）
def process_ok_images():
    for batch_images in load_images_from_directory(ok_dir):
        process_images(batch_images, "OK", is_ng=False)

# NG画像の処理（省略可能な部分）
def process_ng_images():
    for subdir in ['porosity', 'dent', 'crack']:
        subdir_path = os.path.join(ng_dir, subdir)
        print(f"Processing NG images in {subdir_path}")
        for batch_images in load_images_from_directory(subdir_path):
            process_images(batch_images, subdir, is_ng=True)

# メイン処理
process_ok_images()
process_ng_images()

print("処理が完了しました。")

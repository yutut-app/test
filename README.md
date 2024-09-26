# 1. ライブラリのインポート
%pip install -r requirements.txt

import os
import cv2
import numpy as np
import gc

# 2. パラメータの設定
# 1px = 0.0625mm
PIXEL_TO_MM = 0.0625
# 切り取り領域のサイズ (3730 x 3640 px)
CROP_WIDTH = 3730
CROP_HEIGHT = 3640
# データのパス
input_data_dir = os.path.join("..", "data", "input")
output_data_dir = os.path.join("..", "data", "output")
work_frame_right_dir = os.path.join(input_data_dir, "work_frame_right")  # 右側の参照画像
work_frame_left_dir = os.path.join(input_data_dir, "work_frame_left")  # 左側の参照画像

# OKとNGのディレクトリ設定
ok_dir = os.path.join(input_data_dir, 'OK')
ng_dir = os.path.join(input_data_dir, 'NG')
output_ok_dir = os.path.join(output_data_dir, 'OK')
output_ng_dir = os.path.join(output_data_dir, 'NG')

# 3. 参照画像の読み込み
def load_reference_images(directory):
    ref_images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 参照画像はグレースケールで読み込む
        if img is not None:
            ref_images.append(img)
    return ref_images

# 右側と左側の参照画像を読み込む
right_reference_images = load_reference_images(work_frame_right_dir)
left_reference_images = load_reference_images(work_frame_left_dir)

# 4. 入力画像のワーク側を判定する関数
def match_work_side(input_image):
    # 右側の参照画像とパターンマッチング
    best_right_match = max([cv2.matchTemplate(input_image, ref, cv2.TM_CCOEFF_NORMED).max() for ref in right_reference_images])
    
    # 左側の参照画像とパターンマッチング
    best_left_match = max([cv2.matchTemplate(input_image, ref, cv2.TM_CCOEFF_NORMED).max() for ref in left_reference_images])
    
    # マッチ率の高い方を返す
    if best_right_match > best_left_match:
        return "right"
    else:
        return "left"

# 5. ワークの部分を切り出す関数
def crop_work_area(image, side):
    h, w = image.shape
    
    if side == "right":
        # 右側の画像なら、左下を基準に切り出し
        return image[h - CROP_HEIGHT:h, 0:CROP_WIDTH]
    elif side == "left":
        # 左側の画像なら、右下を基準に切り出し
        return image[h - CROP_HEIGHT:h, w - CROP_WIDTH:w]

# 6. データの読み込みと処理
def load_images_from_directory(directory):
    filenames = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    for filename in filenames:
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 画像はグレースケールで読み込む
        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            continue
        yield filename, img
        gc.collect()  # メモリ解放

# 7. OK画像とNG画像の処理
def process_ok_images():
    for filename, image in load_images_from_directory(ok_dir):
        process_image(filename, image, is_ng=False)

def process_ng_images():
    for defect_type in ['porosity', 'dent', 'crack']:
        defect_dir = os.path.join(ng_dir, defect_type)
        for filename, image in load_images_from_directory(defect_dir):
            process_image(filename, image, defect_type, is_ng=True)

# 8. 画像を処理する関数
def process_image(filename, image, defect_type=None, is_ng=False):
    # ワークのどちら側かを判定
    work_side = match_work_side(image)
    print(f"{filename}: Detected work side: {work_side}")
    
    # ワーク部分を切り出し
    cropped_image = crop_work_area(image, work_side)
    
    # 切り出し後の画像を保存
    if is_ng:
        save_dir = os.path.join(output_ng_dir, defect_type)
    else:
        save_dir = output_ok_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f"cropped_{filename}")
    cv2.imwrite(save_path, cropped_image)
    print(f"Saved cropped image: {save_path}")

# OK画像の処理
process_ok_images()

# NG画像の処理
process_ng_images()

print("処理が完了しました。")

# 1. ライブラリのインポート
%pip install -r requirements.txt

import os
import cv2
import numpy as np
import gc  # ガベージコレクションを利用するためのモジュール

# 2. パラメータの設定
# 1px = 0.0625mm
PIXEL_TO_MM = 0.0625
# 切り取る範囲のサイズ (3730 x 3640 px) -> mmへの換算は必要な場合に考慮
CUT_WIDTH = 3730
CUT_HEIGHT = 3640

# データのパス
input_data_dir = os.path.join("..", "data", "input")
output_data_dir = os.path.join("..", "data", "output")
template_right_dir = os.path.join(input_data_dir, 'work_frame_right')  # 右側ワークの参照画像
template_left_dir = os.path.join(input_data_dir, 'work_frame_left')    # 左側ワークの参照画像

# OKとNGのディレクトリ設定
ok_dir = os.path.join(input_data_dir, 'OK')
ng_dir = os.path.join(input_data_dir, 'NG')
output_ok_dir = os.path.join(output_data_dir, 'OK')
output_ng_dir = os.path.join(output_data_dir, 'NG')

# 必要なディレクトリを作成（存在しない場合）
os.makedirs(output_ok_dir, exist_ok=True)
os.makedirs(output_ng_dir, exist_ok=True)

# 3. テンプレート画像の読み込み
def load_template_images():
    right_template = []
    left_template = []

    for filename in os.listdir(template_right_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(template_right_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                right_template.append(img)

    for filename in os.listdir(template_left_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(template_left_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                left_template.append(img)

    return right_template, left_template

# 4. テンプレートマッチングによる右側か左側かの認識
def recognize_work_side(input_img, right_template, left_template):
    # 入力画像と右側、左側のテンプレート画像を比較して、どちらに近いかを判断
    right_match = 0
    left_match = 0

    # 右側テンプレートとの比較
    for template in right_template:
        res = cv2.matchTemplate(input_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        right_match = max(right_match, max_val)

    # 左側テンプレートとの比較
    for template in left_template:
        res = cv2.matchTemplate(input_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        left_match = max(left_match, max_val)

    # 最大値を比較して、右側か左側かを判断
    return 'right' if right_match > left_match else 'left'

# 5. ワーク部分の切り出し
def crop_work_region(image, work_side):
    height, width = image.shape

    if work_side == 'right':
        # 右側なら左下を基準として3730x3640を切り取り
        cropped_image = image[height - CUT_HEIGHT:height, 0:CUT_WIDTH]
    else:
        # 左側なら右下を基準として3730x3640を切り取り
        cropped_image = image[height - CUT_HEIGHT:height, width - CUT_WIDTH:width]

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

# 7. データの読み込みとワーク部分の切り出し処理
def process_images():
    right_template, left_template = load_template_images()

    # OK画像の処理
    for filename in os.listdir(ok_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(ok_dir, filename)
            input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if input_img is None:
                print(f"Warning: Failed to load image {img_path}")
                continue

            # ワークの左右を判定
            work_side = recognize_work_side(input_img, right_template, left_template)
            cropped_image = crop_work_region(input_img, work_side)
            save_cropped_image(cropped_image, filename, "OK", is_ng=False)

    # NG画像の処理
    for defect_type in ['porosity', 'dent', 'crack']:
        defect_dir = os.path.join(ng_dir, defect_type)
        for filename in os.listdir(defect_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(defect_dir, filename)
                input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if input_img is None:
                    print(f"Warning: Failed to load image {img_path}")
                    continue

                # ワークの左右を判定
                work_side = recognize_work_side(input_img, right_template, left_template)
                cropped_image = crop_work_region(input_img, work_side)
                save_cropped_image(cropped_image, filename, defect_type, is_ng=True)

# メイン処理の実行
process_images()

print("処理が完了しました。")


import os
import cv2
import numpy as np
from skimage import io, filters, feature, measure
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import pandas as pd
import gc

# バッチサイズの設定
BATCH_SIZE = 5  # バッチサイズを小さくしています

# ディレクトリとファイルパス
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"
template_dir = os.path.join(input_data_dir, "template")
right_template_path = os.path.join(template_dir, "right_keyence.jpg")
left_template_path = os.path.join(template_dir, "left_keyence.jpg")

# ラベル定義
ng_labels = ['label1', 'label2', 'label3']  # label1: 鋳巣, label2: 凹み, label3: 亀裂

# 画像処理パラメータ
crop_width = 1360  # ワーク接合部を削除するための幅
threshold_value = 150  # 二値化しきい値
kernel_size = (5, 5)  # カーネルサイズ
iterations_open = 3  # 膨張処理の繰り返し回数
iterations_close = 20  # 収縮処理の繰り返し回数
gaussian_kernel_size = (7, 7)  # ガウシアンブラーのカーネルサイズ
canny_min_threshold = 30  # エッジ検出の最小しきい値
canny_max_threshold = 120  # エッジ検出の最大しきい値
sigma = 3  # ガウシアンブラーの標準偏差

# 欠陥サイズパラメータ
min_defect_size = 5  # 最小欠陥サイズ（0.5mm = 5px）
max_defect_size = 100  # 最大欠陥サイズ（10mm = 100px）

# テクスチャ検出パラメータ
texture_threshold = 15  # テクスチャの変化を検出するためのしきい値

# エッジ補完のパラメータ
edge_kernel_size = (3, 3)  # エッジ補完のカーネルサイズ
edge_open_iterations = 2   # ノイズ削除の繰り返し回数
edge_close_iterations = 2  # エッジ補完の繰り返し回数

# マスクエッジ検出のパラメータ
mask_edge_min_threshold = 100
mask_edge_max_threshold = 200
mask_edge_margin = 5  # マスクエッジの余裕幅（ピクセル単位）

# 欠陥候補の保存パラメータ
enlargement_factor = 10  # 欠陥候補画像の拡大倍率

def load_origin_keyence_images(directory):
    normal_images = {}
    shape_images = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "Normal" in file and file.endswith(".jpg"):
                base_name = file.replace("Normal", "")
                normal_images[base_name] = os.path.join(root, file)
            elif "Shape" in file and file.endswith(".jpg"):
                base_name = file.replace("Shape", "")
                shape_images[base_name] = os.path.join(root, file)
    
    matched_images = []
    for base_name in normal_images:
        if base_name in shape_images:
            matched_images.append((normal_images[base_name], shape_images[base_name]))
    return matched_images

def template_matching(image, template_path):
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc

def remove_joint_part(image_path, keyence_image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keyence_image = cv2.imread(keyence_image_path, cv2.IMREAD_GRAYSCALE)
    
    right_val, _ = template_matching(keyence_image, right_template_path)
    left_val, _ = template_matching(keyence_image, left_template_path)
    
    if right_val > left_val:
        cropped_image = image[:, :-crop_width]
        cropped_keyence_image = keyence_image[:, :-crop_width]
    else:
        cropped_image = image[:, crop_width:]
        cropped_keyence_image = keyence_image[:, crop_width:]
    
    return cropped_image, cropped_keyence_image

def process_images_with_progress(image_paths, process_func, description):
    total = len(image_paths)
    results = []
    for i in range(0, total, BATCH_SIZE):
        batch = image_paths[i:i+BATCH_SIZE]
        batch_results = process_func(batch)
        results.extend(batch_results)
        print(f"\r{description}: {min(i+BATCH_SIZE, total)}/{total} ({min((i+BATCH_SIZE)/total*100, 100):.2f}%)", end="", flush=True)
        # メモリを解放
        del batch_results
        gc.collect()
    print()  # 改行を追加
    return results

def load_and_remove_joint_part(image_pairs):
    updated_images = []
    for origin_image_path, keyence_image_path in image_pairs:
        cropped_image, cropped_keyence_image = remove_joint_part(origin_image_path, keyence_image_path)
        updated_images.append((cropped_image, cropped_keyence_image))
    return updated_images

ng_images_label1 = process_images_with_progress(
    load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label1")),
    load_and_remove_joint_part,
    "Processing NG Label 1"
)
ng_images_label2 = process_images_with_progress(
    load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label2")),
    load_and_remove_joint_part,
    "Processing NG Label 2"
)
ng_images_label3 = process_images_with_progress(
    load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label3")),
    load_and_remove_joint_part,
    "Processing NG Label 3"
)
ok_images = process_images_with_progress(
    load_origin_keyence_images(os.path.join(input_data_dir, "OK")),
    load_and_remove_joint_part,
    "Processing OK images"
)


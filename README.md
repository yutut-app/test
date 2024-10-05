# app.py

# 必要なライブラリのインポート
import os
import cv2
import numpy as np
import streamlit as st
from skimage import io, filters, feature, measure
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Streamlitの設定
st.title("画像処理パラメータ設定アプリ")

# 1. ライブラリのインポート
st.header("1. ライブラリのインポート")
st.write("必要なライブラリをインポートします。")

# 2. パラメータの設定
st.header("2. パラメータの設定")
st.write("画像処理に使用するパラメータを設定します。")

# ディレクトリとファイルパス
input_data_dir = st.text_input("入力データディレクトリ", "../data/input")
output_data_dir = st.text_input("出力データディレクトリ", "../data/output")
template_dir = os.path.join(input_data_dir, "template")
right_template_path = os.path.join(template_dir, "right_keyence.jpg")
left_template_path = os.path.join(template_dir, "left_keyence.jpg")

# ラベル定義
ng_labels = st.multiselect("NGラベルの選択", ['label1', 'label2', 'label3'], ['label1', 'label2', 'label3'])

# 画像処理パラメータ
st.subheader("画像処理パラメータ")
crop_width = st.number_input("ワーク接合部を削除するための幅", min_value=0, value=1360)
threshold_value = st.slider("二値化しきい値", min_value=0, max_value=255, value=150)
kernel_size_value = st.number_input("カーネルサイズ（単一の整数）", min_value=1, value=5)
kernel_size = (kernel_size_value, kernel_size_value)
iterations_open = st.number_input("膨張処理の繰り返し回数", min_value=1, value=3)
iterations_close = st.number_input("収縮処理の繰り返し回数", min_value=1, value=20)
gaussian_kernel_size_value = st.number_input("ガウシアンブラーのカーネルサイズ（単一の整数）", min_value=1, value=7)
gaussian_kernel_size = (gaussian_kernel_size_value, gaussian_kernel_size_value)
canny_min_threshold = st.slider("エッジ検出の最小しきい値", min_value=0, max_value=255, value=30)
canny_max_threshold = st.slider("エッジ検出の最大しきい値", min_value=0, max_value=255, value=120)
sigma = st.number_input("ガウシアンブラーの標準偏差", min_value=0.0, value=3.0)

# 欠陥サイズパラメータ
st.subheader("欠陥サイズパラメータ")
min_defect_size = st.number_input("最小欠陥サイズ（ピクセル）", min_value=1, value=5)
max_defect_size = st.number_input("最大欠陥サイズ（ピクセル）", min_value=1, value=100)

# テクスチャ検出パラメータ
st.subheader("テクスチャ検出パラメータ")
texture_threshold = st.slider("テクスチャの変化を検出するためのしきい値", min_value=0, max_value=255, value=15)

# エッジ補完のパラメータ
st.subheader("エッジ補完のパラメータ")
edge_close_kernel_size_value = st.number_input("エッジ補完のカーネルサイズ（単一の整数）", min_value=1, value=3)
edge_close_kernel_size = (edge_close_kernel_size_value, edge_close_kernel_size_value)
edge_close_iterations = st.number_input("エッジ補完の繰り返し回数", min_value=1, value=2)

# マスクエッジ検出のパラメータ
st.subheader("マスクエッジ検出のパラメータ")
mask_edge_min_threshold = st.slider("マスクエッジ検出の最小しきい値", min_value=0, max_value=255, value=100)
mask_edge_max_threshold = st.slider("マスクエッジ検出の最大しきい値", min_value=0, max_value=255, value=200)
mask_edge_margin = st.number_input("マスクエッジの余裕幅（ピクセル）", min_value=0, value=5)

# 欠陥候補の保存パラメータ
st.subheader("欠陥候補の保存パラメータ")
enlargement_factor = st.number_input("欠陥候補画像の拡大倍率", min_value=1, value=10)

# 3. データの読み込み
st.header("3. データの読み込み")
st.write("入力画像をアップロードします。")

uploaded_files = st.file_uploader("入力画像を選択してください（複数選択可）", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

# アップロードされた画像を表示
if uploaded_files:
    st.subheader("アップロードされた画像")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_column_width=True)

# 画像処理を開始する
if st.button("画像処理を開始"):
    if not uploaded_files:
        st.warning("画像がアップロードされていません。")
    else:
        st.write("画像処理を開始します...")
        # アップロードされた画像を処理用のリストに格納
        image_pairs = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            # ここでは、キーエンス画像として同じ画像を使用（実際には別のキーエンス画像を使用する必要があります）
            image_pairs.append((image_np, image_np))

        # 4. ワーク接合部の削除
        st.header("4. ワーク接合部の削除")
        def remove_joint_part(image, crop_width):
            # ワークの接合部を削除する関数
            cropped_image = image[:, crop_width:]
            return cropped_image

        processed_images = []
        for i, (origin_image, keyence_image) in enumerate(image_pairs):
            cropped_image = remove_joint_part(origin_image, crop_width)
            processed_images.append((cropped_image, keyence_image))

            st.subheader(f"ワーク接合部削除後の画像 {i+1}")
            st.image(cropped_image, caption=f"接合部削除後の画像 {i+1}", use_column_width=True)

        # 5. 二値化によるマスクの作成
        st.header("5. 二値化によるマスクの作成")
        def binarize_image(image):
            # 二値化とマスク作成
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)

            return binary_image

        binarized_images = []
        for i, (cropped_image, keyence_image) in enumerate(processed_images):
            binarized_image = binarize_image(cropped_image)
            binarized_images.append((binarized_image, keyence_image))

            st.subheader(f"二値化マスク画像 {i+1}")
            st.image(binarized_image, caption=f"二値化マスク画像 {i+1}", use_column_width=True)

        # 6. エッジ検出とテクスチャ検出の改良
        st.header("6. エッジ検出とテクスチャ検出の改良")
        def detect_edges_and_texture(cropped_keyence_image, binarized_image):
            # エッジ検出とテクスチャ検出
            masked_image = cv2.bitwise_and(cropped_keyence_image, cropped_keyence_image, mask=binarized_image)
            blurred_image = cv2.GaussianBlur(masked_image, gaussian_kernel_size, sigma)
            edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
            laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
            abs_laplacian = np.absolute(laplacian)
            laplacian_edges = np.uint8(abs_laplacian > texture_threshold) * 255
            combined_edges = cv2.bitwise_or(edges, laplacian_edges)
            return combined_edges

        edged_images = []
        for i, (binarized_image, keyence_image) in enumerate(binarized_images):
            edge_image = detect_edges_and_texture(keyence_image, binarized_image)
            edged_images.append((binarized_image, edge_image))

            st.subheader(f"エッジ検出画像 {i+1}")
            st.image(edge_image, caption=f"エッジ検出画像 {i+1}", use_column_width=True)

        # 7. エッジの補完とラベリング処理
        st.header("7. エッジの補完とラベリング処理")
        def create_mask_edge_margin(mask, margin):
            # マスクエッジの余裕を持たせる
            mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
            kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
            dilated_edges = cv2.dilate(mask_edges, kernel, iterations=1)
            return dilated_edges

        def complete_edges(edge_image, mask):
            # エッジの補完
            mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
            skeleton = skeletonize(edge_image > 0)
            kernel = np.ones(edge_close_kernel_size, np.uint8)
            connected_skeleton = cv2.morphologyEx(skeleton.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=edge_close_iterations)
            completed_edges = np.maximum(edge_image, connected_skeleton * 255)
            completed_edges = np.where(mask_edges_with_margin > 0, edge_image, completed_edges)
            return completed_edges.astype(np.uint8)

        def label_and_measure_defects(edge_image):
            # 欠陥候補のラベリングと特徴量の測定
            binary_edge_image = (edge_image > 0).astype(np.uint8)
            labels = measure.label(binary_edge_image, connectivity=2)
            defects = []
            for region in measure.regionprops(labels):
                y, x = region.bbox[0], region.bbox[1]
                h, w = region.bbox[2] - y, region.bbox[3] - x
                defect_info = {
                    'label': region.label,
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'area': region.area,
                    'centroid_y': region.centroid[0], 'centroid_x': region.centroid[1],
                    'perimeter': region.perimeter,
                    'eccentricity': region.eccentricity,
                    'orientation': region.orientation,
                    'major_axis_length': region.major_axis_length,
                    'minor_axis_length': region.minor_axis_length,
                    'solidity': region.solidity,
                    'extent': region.extent,
                    'aspect_ratio': max(w, h) / min(w, h) if min(w, h) > 0 else 0,
                    'max_length': max(w, h)
                }
                defects.append(defect_info)
            return defects

        labeled_images = []
        for i, (binarized_image, edge_image) in enumerate(edged_images):
            completed_edges = complete_edges(edge_image, binarized_image)
            defects = label_and_measure_defects(completed_edges)
            labeled_images.append((binarized_image, completed_edges, defects))

            # 欠陥候補を表示
            st.subheader(f"ラベリング結果 {i+1}")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(completed_edges, cmap='gray')
            for defect in defects:
                rect = plt.Rectangle((defect['x'], defect['y']), defect['width'], defect['height'],
                                     fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                ax.text(defect['x'], defect['y'], str(defect['label']), color='red', fontsize=12)
            plt.axis('off')
            st.pyplot(fig)

        # 8. 欠陥候補のフィルタリング
        st.header("8. 欠陥候補のフィルタリング")
        def remove_defects_on_mask_edge(defects, mask):
            mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
            filtered_defects = []
            for defect in defects:
                x, y, w, h = defect['x'], defect['y'], defect['width'], defect['height']
                if not np.any(mask_edges[y:y+h, x:x+w] > 0):
                    filtered_defects.append(defect)
            return filtered_defects

        def filter_defects_by_max_length(defects, min_size, max_size):
            return [defect for defect in defects if min_size <= defect['max_length'] <= max_size]

        filtered_images = []
        for i, (binarized_image, edge_image, defects) in enumerate(labeled_images):
            filtered_defects = remove_defects_on_mask_edge(defects, binarized_image)
            filtered_defects = filter_defects_by_max_length(filtered_defects, min_defect_size, max_defect_size)

            for j, defect in enumerate(filtered_defects, 1):
                defect['label'] = j

            filtered_images.append((binarized_image, edge_image, filtered_defects))

            # フィルタリング結果を表示
            st.subheader(f"フィルタリング結果 {i+1}")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(edge_image, cmap='gray')
            for defect in filtered_defects:
                rect = plt.Rectangle((defect['x'], defect['y']), defect['width'], defect['height'],
                                     fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                ax.text(defect['x'], defect['y'], str(defect['label']), color='red', fontsize=12)
            plt.axis('off')
            st.pyplot(fig)

        # 9. 欠陥候補の画像の保存とCSV出力
        st.header("9. 欠陥候補の画像の保存とCSV出力")
        def save_defect_image(image, defect, output_dir, image_name, defect_number):
            cx, cy = defect['centroid_x'], defect['centroid_y']
            size = max(defect['width'], defect['height'])

            x1 = max(int(cx - size), 0)
            y1 = max(int(cy - size), 0)
            x2 = min(int(cx + size), image.shape[1])
            y2 = min(int(cy + size), image.shape[0])

            defect_image = image[y1:y2, x1:x2]
            enlarged_image = cv2.resize(defect_image, (0, 0), fx=enlargement_factor, fy=enlargement_factor)

            output_filename = f"defect_{defect_number}.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, enlarged_image)

            return output_filename

        def process_images_for_saving(filtered_images, base_output_dir):
            all_defects_data = []

            for i, (binarized_image, edge_image, defects) in enumerate(filtered_images):
                image_name = f"image_{i+1}"
                output_dir = os.path.join(base_output_dir, image_name)
                os.makedirs(output_dir, exist_ok=True)

                for defect in defects:
                    output_filename = save_defect_image(edge_image, defect, output_dir, image_name, defect['label'])

                    defect_data = {
                        'image_name': image_name,
                        'defect_image': os.path.join(image_name, output_filename),
                        'Image_label': 1,  # NG画像として扱う
                        'defect_label': 0,  # デフォルトで0（OK）とする
                    }
                    defect_data.update(defect)
                    all_defects_data.append(defect_data)

            return all_defects_data

        output_dir = os.path.join(output_data_dir, "defect_images")
        os.makedirs(output_dir, exist_ok=True)

        all_defects_data = process_images_for_saving(filtered_images, output_dir)

        # CSVファイルに出力
        df = pd.DataFrame(all_defects_data)
        csv_output_path = os.path.join(output_data_dir, "defects_data.csv")
        df.to_csv(csv_output_path, index=False)

        st.write(f"欠陥候補の画像とデータを保存しました。CSVファイルのパス: {csv_output_path}")

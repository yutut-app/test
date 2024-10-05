import streamlit as st
import cv2
import numpy as np
from skimage import io, filters, feature, measure, morphology
import matplotlib.pyplot as plt
from PIL import Image
import io as io_lib

# Streamlitの設定
st.title("画像処理パイプラインデモ")
st.sidebar.title("パラメータ設定")

# 1. ライブラリのインポート
import os
import pandas as pd

# 2. パラメータの設定（サイドバーで設定可能）

st.sidebar.header("画像処理パラメータ設定")

# 画像処理パラメータ
crop_width = st.sidebar.number_input("ワーク接合部の削除幅（ピクセル）", value=1360, step=10, min_value=0)
threshold_value = st.sidebar.slider("二値化しきい値", min_value=0, max_value=255, value=150)
kernel_size = st.sidebar.slider("カーネルサイズ（膨張・収縮）", min_value=1, max_value=15, value=5, step=2)
iterations_open = st.sidebar.number_input("オープン処理の繰り返し回数", value=3, min_value=1)
iterations_close = st.sidebar.number_input("クローズ処理の繰り返し回数", value=20, min_value=1)
gaussian_kernel_size = st.sidebar.slider("ガウシアンブラーのカーネルサイズ", min_value=1, max_value=15, value=7, step=2)
canny_min_threshold = st.sidebar.slider("Cannyエッジ検出の最小しきい値", min_value=0, max_value=255, value=30)
canny_max_threshold = st.sidebar.slider("Cannyエッジ検出の最大しきい値", min_value=0, max_value=255, value=120)
sigma = st.sidebar.slider("ガウシアンブラーの標準偏差", min_value=0, max_value=10, value=3)
min_defect_size = st.sidebar.number_input("最小欠陥サイズ（ピクセル）", value=5, min_value=1)
max_defect_size = st.sidebar.number_input("最大欠陥サイズ（ピクセル）", value=100, min_value=1)
texture_threshold = st.sidebar.slider("テクスチャ検出のしきい値", min_value=0, max_value=255, value=15)
edge_margin = st.sidebar.number_input("マスクエッジの余裕幅（ピクセル）", value=5, min_value=0)

# 3. 画像のアップロード
st.sidebar.header("入力画像のアップロード")
uploaded_origin_image = st.sidebar.file_uploader("元画像を選択", type=["jpg", "png", "bmp"])
uploaded_keyence_image = st.sidebar.file_uploader("キーエンス前処理画像を選択", type=["jpg", "png", "bmp"])

if uploaded_origin_image is not None and uploaded_keyence_image is not None:
    # 画像の読み込み
    origin_image = Image.open(uploaded_origin_image)
    keyence_image = Image.open(uploaded_keyence_image)

    origin_image_np = np.array(origin_image.convert('RGB'))
    keyence_image_np = np.array(keyence_image.convert('RGB'))

    st.image([origin_image, keyence_image], caption=["元画像", "キーエンス前処理画像"], width=300)

    # 4. ワーク接合部の削除
    st.header("4. ワーク接合部の削除")

    def remove_joint_part(image, keyence_image, crop_width):
        # 左右判定のためのテンプレートマッチングは省略（画像が単体のため）
        # 本コードでは右側を削除する例として実装
        cropped_image = image[:, :-crop_width]
        cropped_keyence_image = keyence_image[:, :-crop_width]
        return cropped_image, cropped_keyence_image

    cropped_image, cropped_keyence_image = remove_joint_part(origin_image_np, keyence_image_np, crop_width)

    st.image([cropped_image, cropped_keyence_image], caption=["接合部削除後の元画像", "接合部削除後のキーエンス画像"], width=300)

    # 5. 二値化によるマスクの作成
    st.header("5. 二値化によるマスクの作成")

    def binarize_image(image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)

        return binary_image

    binarized_image = binarize_image(cropped_image)

    st.image(binarized_image, caption="二値化マスク", width=300)

    # 6. エッジ検出とテクスチャ検出の改良
    st.header("6. エッジ検出とテクスチャ検出の改良")

    def detect_edges_and_texture(cropped_keyence_image, binarized_image):
        masked_image = cv2.bitwise_and(cropped_keyence_image, cropped_keyence_image, mask=binarized_image)
        blurred_image = cv2.GaussianBlur(masked_image, (gaussian_kernel_size, gaussian_kernel_size), sigma)
        gray_blurred = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_blurred, canny_min_threshold, canny_max_threshold)
        laplacian = cv2.Laplacian(gray_blurred, cv2.CV_64F)
        abs_laplacian = np.absolute(laplacian)
        laplacian_edges = np.uint8(abs_laplacian > texture_threshold) * 255
        combined_edges = cv2.bitwise_or(edges, laplacian_edges)
        return combined_edges

    edge_image = detect_edges_and_texture(cropped_keyence_image, binarized_image)

    st.image(edge_image, caption="エッジ検出結果", width=300)

    # 7. ラベリング処理
    st.header("7. ラベリング処理")

    def remove_defects_on_mask_edge_with_margin(defects, mask, margin):
        mask_edges = cv2.Canny(mask, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * margin + 1, 2 * margin + 1))
        expanded_edges = cv2.dilate(mask_edges, kernel)

        filtered_defects = []
        for (x, y, w, h, cx, cy) in defects:
            if np.any(expanded_edges[y:y+h, x:x+w] > 0):
                continue
            filtered_defects.append((x, y, w, h, cx, cy))
        return filtered_defects

    def label_and_filter_defects_with_margin(edge_image, binarized_image, min_size, max_size, margin):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge_image)

        defects = []
        for i in range(1, num_labels):
            size = stats[i, cv2.CC_STAT_AREA]
            if min_size <= size <= max_size:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]
                defects.append((x, y, w, h, cx, cy))

        filtered_defects = remove_defects_on_mask_edge_with_margin(defects, binarized_image, margin)

        return filtered_defects

    defects = label_and_filter_defects_with_margin(edge_image, binarized_image, min_defect_size, max_defect_size, edge_margin)

    # 8. 欠陥候補の表示
    st.header("8. 欠陥候補の表示")

    def draw_defects(image, defects):
        result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for (x, y, w, h, cx, cy) in defects:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(result_image, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        return result_image

    result_image = draw_defects(edge_image, defects)

    st.image(result_image, caption="欠陥候補の検出結果", width=300)

else:
    st.info("元画像とキーエンス前処理画像をアップロードしてください。")

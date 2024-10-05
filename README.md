import os
import cv2
import numpy as np
import streamlit as st
from skimage import io, filters, feature, measure
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from PIL import Image

# 1. Streamlitアプリの設定
st.title("画像処理パラメータ設定")

# 2. パラメータの設定 (サイドバー)
st.sidebar.header("パラメータ設定")

# 画像処理パラメータ
st.sidebar.subheader("2. パラメータの設定")

threshold_value = st.sidebar.slider("二値化しきい値", min_value=0, max_value=255, value=150)
kernel_size_value = st.sidebar.number_input("カーネルサイズ（単一の整数）", min_value=1, value=5)
kernel_size = (kernel_size_value, kernel_size_value)
iterations_open = st.sidebar.number_input("膨張処理の繰り返し回数", min_value=1, value=3)
iterations_close = st.sidebar.number_input("収縮処理の繰り返し回数", min_value=1, value=20)
gaussian_kernel_size_value = st.sidebar.number_input("ガウシアンブラーのカーネルサイズ（単一の整数）", min_value=1, value=7)
gaussian_kernel_size = (gaussian_kernel_size_value, gaussian_kernel_size_value)
canny_min_threshold = st.sidebar.slider("エッジ検出の最小しきい値", min_value=0, max_value=255, value=30)
canny_max_threshold = st.sidebar.slider("エッジ検出の最大しきい値", min_value=0, max_value=255, value=120)
sigma = st.sidebar.number_input("ガウシアンブラーの標準偏差", min_value=0.0, value=3.0)

# 欠陥サイズパラメータ
min_defect_size = st.sidebar.number_input("最小欠陥サイズ（ピクセル）", min_value=1, value=5)
max_defect_size = st.sidebar.number_input("最大欠陥サイズ（ピクセル）", min_value=1, value=100)

# エッジ補完のパラメータ
edge_close_kernel_size_value = st.sidebar.number_input("エッジ補完のカーネルサイズ（単一の整数）", min_value=1, value=3)
edge_close_kernel_size = (edge_close_kernel_size_value, edge_close_kernel_size_value)
edge_close_iterations = st.sidebar.number_input("エッジ補完の繰り返し回数", min_value=1, value=2)

# マスクエッジ検出のパラメータ
mask_edge_min_threshold = st.sidebar.slider("マスクエッジ検出の最小しきい値", min_value=0, max_value=255, value=100)
mask_edge_max_threshold = st.sidebar.slider("マスクエッジ検出の最大しきい値", min_value=0, max_value=255, value=200)
mask_edge_margin = st.sidebar.number_input("マスクエッジの余裕幅（ピクセル）", min_value=0, value=5)

# 3. 画像のアップロード
st.subheader("3. 画像のアップロード")
uploaded_image = st.file_uploader("画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # PILで画像を読み込み、表示
    image = Image.open(uploaded_image)
    st.image(image, caption="アップロードされた画像", use_column_width=True)
    
    # OpenCV用に画像を変換
    image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 4. 二値化によるマスクの作成
    st.subheader("4. 二値化によるマスクの作成")
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
    
    st.image(binary_image, caption="二値化された画像", use_column_width=True)
    
    # 5. エッジ検出
    st.subheader("5. エッジ検出")
    blurred_image = cv2.GaussianBlur(gray_image, gaussian_kernel_size, sigma)
    edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
    st.image(edges, caption="エッジ検出結果", use_column_width=True)

    # 6. 欠陥検出とフィルタリング
    st.subheader("6. 欠陥検出とフィルタリング")
    labels, stats, centroids = cv2.connectedComponentsWithStats(edges)
    filtered_defects = []
    
    for i in range(1, labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_defect_size <= area <= max_defect_size:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            filtered_defects.append((x, y, w, h))
    
    defect_image = image.copy()
    for (x, y, w, h) in filtered_defects:
        cv2.rectangle(defect_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    st.image(defect_image, caption="フィルタリングされた欠陥結果", use_column_width=True)

    # 7. 欠陥候補の表示
    st.subheader("7. 欠陥候補の表示")
    for i, (x, y, w, h) in enumerate(filtered_defects, start=1):
        defect_candidate = gray_image[y:y+h, x:x+w]
        enlarged_defect = cv2.resize(defect_candidate, (w*10, h*10), interpolation=cv2.INTER_LINEAR)
        st.image(enlarged_defect, caption=f"欠陥候補 {i}", use_column_width=True)

else:
    st.info("画像をアップロードしてください。")


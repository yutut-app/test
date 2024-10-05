import streamlit as st
import cv2
import numpy as np
from skimage import io, filters, feature, measure
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from io import BytesIO
import PIL.Image

# サイドバーで設定するパラメータ
st.sidebar.title("画像処理パラメータ設定")

# ワーク接合部の削除
crop_width = st.sidebar.slider('ワーク接合部の削除: 幅', min_value=500, max_value=2000, value=1360)

# 二値化によるマスクの作成
threshold_value = st.sidebar.slider('二値化しきい値', min_value=0, max_value=255, value=150)
kernel_size = st.sidebar.slider('カーネルサイズ', min_value=1, max_value=10, value=5)
iterations_open = st.sidebar.slider('膨張処理の繰り返し回数', min_value=1, max_value=10, value=3)
iterations_close = st.sidebar.slider('収縮処理の繰り返し回数', min_value=1, max_value=30, value=20)

# エッジ検出とテクスチャ検出
gaussian_kernel_size = st.sidebar.slider('ガウシアンブラーのカーネルサイズ', min_value=3, max_value=15, value=7, step=2)
sigma = st.sidebar.slider('ガウシアンブラーの標準偏差', min_value=0.1, max_value=10.0, value=3.0)
canny_min_threshold = st.sidebar.slider('エッジ検出の最小しきい値', min_value=0, max_value=255, value=30)
canny_max_threshold = st.sidebar.slider('エッジ検出の最大しきい値', min_value=0, max_value=255, value=120)
texture_threshold = st.sidebar.slider('テクスチャの変化を検出するためのしきい値', min_value=0, max_value=50, value=15)

# エッジ補完
edge_kernel_size = st.sidebar.slider('エッジ補完のカーネルサイズ', min_value=1, max_value=10, value=3)
edge_open_iterations = st.sidebar.slider('ノイズ削除の繰り返し回数', min_value=1, max_value=10, value=2)
edge_close_iterations = st.sidebar.slider('エッジ補完の繰り返し回数', min_value=1, max_value=10, value=2)

# マスクエッジ検出
mask_edge_min_threshold = st.sidebar.slider('マスクエッジ最小しきい値', min_value=0, max_value=255, value=100)
mask_edge_max_threshold = st.sidebar.slider('マスクエッジ最大しきい値', min_value=0, max_value=255, value=200)
mask_edge_margin = st.sidebar.slider('マスクエッジの余裕幅（ピクセル単位）', min_value=0, max_value=20, value=5)

# 欠陥サイズのフィルタリング
min_defect_size = st.sidebar.slider('最小欠陥サイズ（ピクセル）', min_value=1, max_value=50, value=5)
max_defect_size = st.sidebar.slider('最大欠陥サイズ（ピクセル）', min_value=50, max_value=200, value=100)

# 画像アップロード（サイドバーではなくメイン画面）
st.title("画像アップロード")
uploaded_original_image = st.file_uploader("元画像をアップロード", type=['jpg', 'png'])
uploaded_keyence_image = st.file_uploader("キーエンス前処理画像をアップロード", type=['jpg', 'png'])

# 画像処理を行う関数
def process_image(original_image, keyence_image):
    # 画像をリサイズして表示
    st.image([original_image, keyence_image], caption=["元画像", "キーエンス前処理画像"], use_column_width=True)
    
    # 1. ワーク接合部の削除
    keyence_image = np.array(keyence_image.convert('RGB'))
    original_image = np.array(original_image.convert('RGB'))
    
    height, width = keyence_image.shape[:2]
    if width > crop_width:
        cropped_original = original_image[:, crop_width:]
        cropped_keyence = keyence_image[:, crop_width:]
    else:
        cropped_original = original_image[:, :width - crop_width]
        cropped_keyence = keyence_image[:, :width - crop_width]
    
    st.image(cropped_original, caption="ワーク接合部を削除後の元画像", use_column_width=True)
    st.image(cropped_keyence, caption="ワーク接合部を削除後のキーエンス前処理画像", use_column_width=True)

    # 2. 二値化によるマスクの作成
    gray_image = cv2.cvtColor(cropped_original, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)

    st.image(binary_image, caption="二値化によるマスク", use_column_width=True)

    # 3. エッジ検出とテクスチャ検出
    blurred_image = cv2.GaussianBlur(cropped_keyence, (gaussian_kernel_size, gaussian_kernel_size), sigma)
    edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
    
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    laplacian_edges = np.uint8(abs_laplacian > texture_threshold) * 255
    combined_edges = cv2.bitwise_or(edges, laplacian_edges)
    
    st.image(combined_edges, caption="エッジとテクスチャの検出結果", use_column_width=True)

    # 4. エッジの補完
    skeleton = skeletonize(combined_edges > 0)
    kernel = np.ones((edge_kernel_size, edge_kernel_size), np.uint8)
    
    opened_skeleton = cv2.morphologyEx(skeleton.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=edge_open_iterations)
    connected_skeleton = cv2.morphologyEx(opened_skeleton, cv2.MORPH_CLOSE, kernel, iterations=edge_close_iterations)
    
    completed_edges = np.maximum(combined_edges, connected_skeleton * 255)
    
    st.image(completed_edges, caption="エッジの補完結果", use_column_width=True)

    # 5. ラベリング処理と欠陥候補の検出
    labels = measure.label(completed_edges, connectivity=2)
    defects = []
    for region in measure.regionprops(labels):
        if min_defect_size <= region.major_axis_length <= max_defect_size:
            defects.append(region)

    # 6. 欠陥候補の切り出しと表示
    cols = st.columns(len(defects))  # 列を欠陥候補の数だけ作成
    for i, defect in enumerate(defects):
        minr, minc, maxr, maxc = defect.bbox
        max_length = max(maxc - minc, maxr - minr)
        
        # 欠陥候補を中心にした正方形の切り出し
        center_y, center_x = defect.centroid
        half_length = max_length // 2
        
        x1 = int(center_x - half_length)
        y1 = int(center_y - half_length)
        x2 = int(center_x + half_length)
        y2 = int(center_y + half_length)
        
        cut_out = cropped_keyence[y1:y2, x1:x2]
        
        # 欠陥候補の表示
        with cols[i]:
            st.image(cut_out, caption=f"欠陥候補 {i+1} (尺度 {max_length}px)", use_column_width=True)

# 画像がアップロードされた場合のみ処理を実行
if uploaded_original_image is not None and uploaded_keyence_image is not None:
    original_image = PIL.Image.open(BytesIO(uploaded_original_image.read()))
    keyence_image = PIL.Image.open(BytesIO(uploaded_keyence_image.read()))
    
    process_image(original_image,

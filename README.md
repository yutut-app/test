import streamlit as st
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

# サイドバーでパラメータを設定
st.sidebar.title("エッジ検出パラメータ設定")

# ガウシアンブラーのカーネルサイズ（奇数のみ）
blur_kernel_size = st.sidebar.slider("ガウシアンブラー カーネルサイズ", 1, 21, 5, step=2)

# Canny エッジ検出の閾値
canny_threshold1 = st.sidebar.slider("Canny エッジ検出 閾値1", 0, 300, 50)
canny_threshold2 = st.sidebar.slider("Canny エッジ検出 閾値2", 0, 300, 150)

# ラベリングの際の連結性
connectivity = st.sidebar.slider("連結性", 1, 2, 2)

# 画像のアップロード
uploaded_file = st.sidebar.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像を読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ガウシアンブラーを適用
    blurred_img = cv2.GaussianBlur(gray_image, (blur_kernel_size, blur_kernel_size), 0)

    # Cannyエッジ検出
    edges = cv2.Canny(blurred_img, canny_threshold1, canny_threshold2)

    # ラベリングによる輪郭抽出
    labeled_image = measure.label(edges, connectivity=connectivity)
    properties = measure.regionprops(labeled_image)

    # ラベリング結果のプロット
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # オリジナルの画像
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("オリジナル画像")
    ax[0].axis("off")

    # エッジ検出結果
    ax[1].imshow(edges, cmap="gray")
    ax[1].set_title("エッジ検出結果")
    ax[1].axis("off")

    st.pyplot(fig)

    # 各ラベル領域をリスト化して表示
    st.write(f"検出された領域数: {len(properties)}")
    for i, prop in enumerate(properties):
        st.write(f"領域 {i + 1}: 面積 = {prop.area}")
else:
    st.write("左のサイドバーから画像をアップロードしてください。")

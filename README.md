import streamlit as st
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

# サイドバーで設定モードを選択
st.sidebar.title("モード選択")
mode = st.sidebar.selectbox("設定する機能を選択してください", ["エッジ検出パラメータ設定", "二値化パラメータ設定"])

# 画像のアップロード
uploaded_file = st.sidebar.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像を読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mode == "エッジ検出パラメータ設定":
        # エッジ検出パラメータ設定
        st.sidebar.title("エッジ検出パラメータ設定")
        
        # ガウシアンブラーのカーネルサイズ（奇数のみ）
        blur_kernel_size = st.sidebar.slider("ガウシアンブラー カーネルサイズ", 1, 21, 5, step=2)
        
        # Canny エッジ検出の閾値
        canny_threshold1 = st.sidebar.slider("Canny エッジ検出 閾値1", 0, 300, 50)
        canny_threshold2 = st.sidebar.slider("Canny エッジ検出 閾値2", 0, 300, 150)
        
        # ガウシアンブラーを適用
        blurred_img = cv2.GaussianBlur(gray_image, (blur_kernel_size, blur_kernel_size), 0)
        
        # Cannyエッジ検出
        edges = cv2.Canny(blurred_img, canny_threshold1, canny_threshold2)
        
        # ラベリングによる輪郭抽出
        labeled_image = measure.label(edges, connectivity=2)
        properties = measure.regionprops(labeled_image)
        
        # 結果の表示
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

    elif mode == "二値化パラメータ設定":
        # 二値化パラメータ設定
        st.sidebar.title("二値化パラメータ設定")

        # 閾値の設定
        threshold_value = st.sidebar.slider("二値化 閾値", 0, 255, 128)
        
        # 閾値処理のタイプ選択
        threshold_type = st.sidebar.selectbox(
            "閾値タイプ", 
            ["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO", "THRESH_TOZERO_INV"]
        )

        # OpenCVの閾値タイプに対応する辞書
        threshold_types = {
            "THRESH_BINARY": cv2.THRESH_BINARY,
            "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
            "THRESH_TRUNC": cv2.THRESH_TRUNC,
            "THRESH_TOZERO": cv2.THRESH_TOZERO,
            "THRESH_TOZERO_INV": cv2.THRESH_TOZERO_INV
        }
        
        # 二値化処理
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, threshold_types[threshold_type])

        # 結果の表示
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # オリジナルの画像
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title("オリジナル画像")
        ax[0].axis("off")

        # 二値化結果
        ax[1].imshow(binary_image, cmap="gray")
        ax[1].set_title("二値化結果")
        ax[1].axis("off")

        st.pyplot(fig)

else:
    st.write("左のサイドバーから画像をアップロードしてください。")

以下に、要求された機能を持つStreamlitアプリケーションのPythonコードを示します。このコードは、画像処理パラメータの最適化を行うためのインタラクティブなUIを提供します。

```python
import streamlit as st
import cv2
import numpy as np
from skimage import measure
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import io
import base64

def main():
    st.title("鋳造部品の欠陥検出システム - パラメータ最適化")

    # 1. 画像のアップロード
    st.header("1. 画像のアップロード")
    normal_image = st.file_uploader("元画像（Normal）をアップロード", type=['jpg', 'png'])
    shape_image = st.file_uploader("キーエンス前処理画像（Shape）をアップロード", type=['jpg', 'png'])

    if normal_image is not None and shape_image is not None:
        normal_img = cv2.imdecode(np.frombuffer(normal_image.read(), np.uint8), 1)
        shape_img = cv2.imdecode(np.frombuffer(shape_image.read(), np.uint8), 1)

        # パラメータの設定
        st.sidebar.header("パラメータ設定")

        # 2. ワーク接合部の削除
        st.header("2. ワーク接合部の削除")
        crop_width = st.sidebar.slider("ワーク接合部を削除するための幅", 0, 2000, 1360)
        
        cropped_normal = normal_img[:, crop_width:]
        cropped_shape = shape_img[:, crop_width:]
        
        st.image(cropped_normal, caption="ワーク接合部削除後の元画像", use_column_width=True)
        st.image(cropped_shape, caption="ワーク接合部削除後のキーエンス前処理画像", use_column_width=True)

        # 3. 二値化によるマスクの作成
        st.header("3. 二値化によるマスクの作成")
        threshold_value = st.sidebar.slider("二値化しきい値", 0, 255, 190)
        kernel_size = st.sidebar.slider("カーネルサイズ", 1, 10, 3)
        iterations_open = st.sidebar.slider("膨張処理の繰り返し回数", 0, 50, 20)
        iterations_close = st.sidebar.slider("収縮処理の繰り返し回数", 0, 50, 20)

        gray_normal = cv2.cvtColor(cropped_normal, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_normal, threshold_value, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)

        st.image(mask, caption="二値化マスク", use_column_width=True)

        # 4. エッジ検出とテクスチャ検出
        st.header("4. エッジ検出とテクスチャ検出")
        gaussian_kernel_size = st.sidebar.slider("ガウシアンブラーのカーネルサイズ", 1, 21, 9, step=2)
        sigma = st.sidebar.slider("ガウシアンブラーの標準偏差", 0.1, 10.0, 3.0, step=0.1)
        canny_min_threshold = st.sidebar.slider("Cannyエッジ検出の最小しきい値", 0, 255, 50)
        canny_max_threshold = st.sidebar.slider("Cannyエッジ検出の最大しきい値", 0, 255, 150)
        texture_threshold = st.sidebar.slider("テクスチャ検出の閾値", 0, 50, 4)

        masked_shape = cv2.bitwise_and(cropped_shape, cropped_shape, mask=mask)
        blurred = cv2.GaussianBlur(masked_shape, (gaussian_kernel_size, gaussian_kernel_size), sigma)
        edges = cv2.Canny(blurred, canny_min_threshold, canny_max_threshold)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        texture = np.uint8(np.absolute(laplacian) > texture_threshold) * 255
        combined_edges = cv2.bitwise_or(edges, texture)

        st.image(combined_edges, caption="エッジ検出とテクスチャ検出結果", use_column_width=True)

        # 5. エッジの補完とラベリング処理
        st.header("5. エッジの補完とラベリング処理")
        edge_close_kernel_size = st.sidebar.slider("エッジ補完のカーネルサイズ", 1, 10, 3)
        edge_close_iterations = st.sidebar.slider("エッジ補完の繰り返し回数", 0, 20, 5)
        mask_edge_min_threshold = st.sidebar.slider("マスクエッジ検出の最小閾値", 0, 255, 50)
        mask_edge_max_threshold = st.sidebar.slider("マスクエッジ検出の最大閾値", 0, 255, 150)
        mask_edge_margin = st.sidebar.slider("マスクエッジの余裕幅", 0, 100, 50)

        mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
        kernel = np.ones((mask_edge_margin*2+1, mask_edge_margin*2+1), np.uint8)
        dilated_mask_edges = cv2.dilate(mask_edges, kernel, iterations=1)

        st.image(dilated_mask_edges, caption="マスクエッジ（余裕幅付き）", use_column_width=True)

        # 6. 欠陥候補のフィルタリング
        st.header("6. 欠陥候補のフィルタリング")
        min_defect_size = st.sidebar.slider("最小欠陥サイズ", 1, 50, 5)
        max_defect_size = st.sidebar.slider("最大欠陥サイズ", 50, 500, 100)

        labels = measure.label(combined_edges)
        regions = measure.regionprops(labels)
        filtered_image = np.zeros_like(combined_edges)

        for region in regions:
            if min_defect_size <= region.area <= max_defect_size:
                filtered_image[labels == region.label] = 255

        st.image(filtered_image, caption="フィルタリング後の欠陥候補", use_column_width=True)

        # 7. 欠陥候補の画像を切り出し
        st.header("7. 欠陥候補の画像の切り出し")
        enlargement_factor = st.sidebar.slider("欠陥候補画像の拡大倍率", 1, 20, 10)

        defect_images = []
        for region in measure.regionprops(measure.label(filtered_image)):
            y, x, y2, x2 = region.bbox
            defect = cropped_shape[y:y2, x:x2]
            enlarged = cv2.resize(defect, (0,0), fx=enlargement_factor, fy=enlargement_factor)
            defect_images.append(enlarged)

        cols = st.columns(3)
        for i, defect_img in enumerate(defect_images):
            cols[i % 3].image(defect_img, caption=f"欠陥候補 {i+1}", use_column_width=True)

if __name__ == "__main__":
    main()
```

このコードは以下の機能を提供します：

1. 元画像とキーエンス前処理画像のアップロード
2. ワーク接合部の削除と結果の表示
3. 二値化によるマスク作成と結果の表示
4. エッジ検出とテクスチャ検出の実行と結果の表示
5. エッジの補完とラベリング処理（マスクエッジの表示）
6. 欠陥候補のフィルタリングと結果の表示
7. 欠陥候補の画像切り出しと表示

各処理段階でパラメータを調整できるUIがサイドバーに用意されており、パラメータの変更がリアルタイムで結果に反映されます。

このアプリケーションを実行するには、必要なライブラリ（streamlit, opencv-python, numpy, scikit-image, matplotlib）をインストールし、このコードを`app.py`として保存した後、コマンドラインで`streamlit run app.py`を実行してください。

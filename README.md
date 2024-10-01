申し訳ありません。エラーの原因を修正し、テンプレート画像のパスも正しく設定します。以下に、修正した完全なコードを示します。

```python
申し訳ありません。ご指摘の内容が完全には反映されていませんでした。以下に、ご要望の内容を反映した完全なコードを示します。

```python
import streamlit as st
import cv2
import numpy as np
from skimage import measure
from skimage.morphology import skeletonize
import os

def load_image(image_file):
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def display_image(image, caption):
    st.image(image, caption=caption, use_column_width=True)

def template_matching(image, template_path):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Template image not found: {template_path}")
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc

def remove_joint_part(image, keyence_image, crop_width, right_template_path, left_template_path):
    right_val, _ = template_matching(keyence_image, right_template_path)
    left_val, _ = template_matching(keyence_image, left_template_path)
    
    if right_val > left_val:
        cropped_image = image[:, crop_width:]
        cropped_keyence_image = keyence_image[:, crop_width:]
    else:
        cropped_image = image[:, :-crop_width]
        cropped_keyence_image = keyence_image[:, :-crop_width]
    
    return cropped_image, cropped_keyence_image

def binarize_image(image, threshold_value, kernel_size, iterations_open, iterations_close):
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
    return binary_image

def detect_edges_and_texture(keyence_image, binarized_image, gaussian_kernel_size, sigma, canny_min_threshold, canny_max_threshold, texture_threshold):
    masked_image = cv2.bitwise_and(keyence_image, keyence_image, mask=binarized_image)
    blurred_image = cv2.GaussianBlur(masked_image, gaussian_kernel_size, sigma)
    edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
    gray_blurred = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_blurred, cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    laplacian_edges = np.uint8(abs_laplacian > texture_threshold) * 255
    combined_edges = cv2.bitwise_or(edges, laplacian_edges)
    return combined_edges

def create_mask_edge_margin(mask, mask_edge_min_threshold, mask_edge_max_threshold, mask_edge_margin):
    mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
    kernel = np.ones((mask_edge_margin * 2 + 1, mask_edge_margin * 2 + 1), np.uint8)
    dilated_edges = cv2.dilate(mask_edges, kernel, iterations=1)
    return dilated_edges

def complete_edges(edge_image, mask, edge_close_kernel_size, edge_close_iterations, mask_edge_min_threshold, mask_edge_max_threshold, mask_edge_margin):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_min_threshold, mask_edge_max_threshold, mask_edge_margin)
    skeleton = skeletonize(edge_image > 0)
    kernel = np.ones(edge_close_kernel_size, np.uint8)
    connected_skeleton = cv2.morphologyEx(skeleton.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=edge_close_iterations)
    completed_edges = np.maximum(edge_image, connected_skeleton * 255)
    completed_edges = np.where(mask_edges_with_margin > 0, edge_image, completed_edges)
    return completed_edges.astype(np.uint8)

def label_and_measure_defects(edge_image):
    binary_edge_image = (edge_image > 0).astype(np.uint8)
    labels = measure.label(binary_edge_image, connectivity=2)
    defects = []
    for region in measure.regionprops(labels):
        y, x = region.bbox[0], region.bbox[1]
        h, w = region.bbox[2] - y, region.bbox[3] - x
        defect_info = {
            'label': region.label,
            'x': x, 'y': y, 'width': w, 'height': h,
            'centroid_y': region.centroid[0], 'centroid_x': region.centroid[1],
            'max_length': max(w, h)
        }
        defects.append(defect_info)
    return defects

def remove_defects_on_mask_edge(defects, mask, mask_edge_min_threshold, mask_edge_max_threshold):
    mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
    filtered_defects = []
    for defect in defects:
        x, y, w, h = defect['x'], defect['y'], defect['width'], defect['height']
        if not np.any(mask_edges[y:y+h, x:x+w] > 0):
            filtered_defects.append(defect)
    return filtered_defects

def filter_defects_by_max_length(defects, min_size, max_size):
    return [defect for defect in defects if min_size <= defect['max_length'] <= max_size]

def process_images_for_filtering(binarized_image, edge_image, defects, min_defect_size, max_defect_size, mask_edge_min_threshold, mask_edge_max_threshold):
    filtered_defects = remove_defects_on_mask_edge(defects, binarized_image, mask_edge_min_threshold, mask_edge_max_threshold)
    filtered_defects = filter_defects_by_max_length(filtered_defects, min_defect_size, max_defect_size)
    
    for j, defect in enumerate(filtered_defects, 1):
        defect['label'] = j
    
    return filtered_defects

def draw_defects(image, defects):
    result_image = image.copy()
    for defect in defects:
        cv2.rectangle(result_image, (defect['x'], defect['y']), 
                      (defect['x'] + defect['width'], defect['y'] + defect['height']), (0, 255, 0), 2)
        cv2.putText(result_image, str(defect['label']), (defect['x'], defect['y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return result_image

def main():
    st.title("鋳造部品の欠陥検出パラメータ最適化")

    normal_image = st.file_uploader("元画像（Normal）をアップロード", type=['jpg', 'png'])
    shape_image = st.file_uploader("キーエンス前処理画像（Shape）をアップロード", type=['jpg', 'png'])

    if normal_image is not None and shape_image is not None:
        normal_img = load_image(normal_image)
        shape_img = load_image(shape_image)

        input_data_dir = "data/input"
        template_dir = os.path.join(input_data_dir, "template")
        right_template_path = os.path.join(template_dir, "right_keyence.jpg")
        left_template_path = os.path.join(template_dir, "left_keyence.jpg")

        st.sidebar.header("パラメータ設定")

        crop_width = st.sidebar.slider("Crop Width", 1000, 2000, 1360)
        threshold_value = st.sidebar.slider("Threshold Value", 0, 255, 190)
        kernel_size = st.sidebar.slider("Kernel Size", 1, 10, 3)
        iterations_open = st.sidebar.slider("Iterations Open", 1, 50, 20)
        iterations_close = st.sidebar.slider("Iterations Close", 1, 50, 20)
        gaussian_kernel_size = st.sidebar.slider("Gaussian Kernel Size", 1, 15, 9, step=2)
        sigma = st.sidebar.slider("Sigma", 1, 10, 3)
        canny_min_threshold = st.sidebar.slider("Canny Min Threshold", 0, 255, 50)
        canny_max_threshold = st.sidebar.slider("Canny Max Threshold", 0, 255, 150)
        texture_threshold = st.sidebar.slider("Texture Threshold", 1, 20, 4)
        edge_close_kernel_size = st.sidebar.slider("Edge Close Kernel Size", 1, 10, 3)
        edge_close_iterations = st.sidebar.slider("Edge Close Iterations", 1, 20, 5)
        mask_edge_min_threshold = st.sidebar.slider("Mask Edge Min Threshold", 0, 255, 50)
        mask_edge_max_threshold = st.sidebar.slider("Mask Edge Max Threshold", 0, 255, 150)
        mask_edge_margin = st.sidebar.slider("Mask Edge Margin", 1, 100, 50)
        min_defect_size = st.sidebar.slider("Min Defect Size", 1, 50, 5)
        max_defect_size = st.sidebar.slider("Max Defect Size", 51, 500, 100)
        enlargement_factor = st.sidebar.slider("Enlargement Factor", 1, 20, 10)

        if st.button("処理を実行"):
            try:
                # ワーク接合部の削除
                cropped_normal, cropped_shape = remove_joint_part(normal_img, shape_img, crop_width, right_template_path, left_template_path)
                st.subheader("ワーク接合部削除後の画像")
                display_image(cropped_normal, "元画像")
                display_image(cropped_shape, "キーエンス前処理画像")

                # 二値化によるマスクの作成
                gray_normal = cv2.cvtColor(cropped_normal, cv2.COLOR_RGB2GRAY)
                mask = binarize_image(gray_normal, threshold_value, (kernel_size, kernel_size), iterations_open, iterations_close)
                st.subheader("マスク画像")
                display_image(mask, "マスク")

                # エッジ検出とテクスチャ検出
                edges = detect_edges_and_texture(cropped_shape, mask, (gaussian_kernel_size, gaussian_kernel_size), sigma, canny_min_threshold, canny_max_threshold, texture_threshold)
                st.subheader("エッジ検出結果")
                display_image(edges, "エッジ")

                # エッジの補完とラベリング処理
                completed_edges = complete_edges(edges, mask, (edge_close_kernel_size, edge_close_kernel_size), edge_close_iterations, mask_edge_min_threshold, mask_edge_max_threshold, mask_edge_margin)
                st.subheader("エッジ補完結果")
                display_image(completed_edges, "補完されたエッジ")

                # 欠陥候補のラベリング
                defects = label_and_measure_defects(completed_edges)

                # 欠陥候補のフィルタリング
                filtered_defects = process_images_for_filtering(mask, completed_edges, defects, min_defect_size, max_defect_size, mask_edge_min_threshold, mask_edge_max_threshold)
                result_image = draw_defects(cropped_shape, filtered_defects)
                st.subheader("欠陥候補")
                display_image(result_image, "検出された欠陥候補")

                # 欠陥候補の画像を切り出し
                st.subheader("切り出された欠陥候補")
                for i, defect in enumerate(filtered_defects):
                    x, y, w, h = defect['x'], defect['y'], defect['width'], defect['height']
                    defect_img = cropped_shape[y:y+h, x:x+w]
                    enlarged_defect = cv2.resize(defect_img, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
                    display_image(enlarged_defect, f"欠陥候補 {defect['label']}")

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                st.error("テンプレート画像が正しく読み込めていない可能性があります。パスを確認してください。")

if __name__ == "__main__":
    main()
```

このコードでは以下の変更を行いました：

1. `label_and_measure_defects` 関数に `max_length` の計算を追加しました。
2. `remove_defects_on_mask_edge` 関数を追加し、マスクのエッジと重なる欠陥候補を除外するようにしました。
3. `filter_defects_by_max_length` 関数を追加し、指定されたサイズ範囲内の欠陥候補のみを抽出するようにしました。
4. `process_images_for_filtering` 関数を修正し、上記の2つのフィルタリング処理を適用するようにしました。
5. メイン処理部分で、ラベリングとフィルタリングのステップを分離し、フィルタリング後の欠陥候補を表示するようにしました。
6. `draw_defects` 関数を修正し、欠陥候補にラベル番号を表示するようにしました。

これらの変更により、ご要望の内容が全て反映されています。欠陥候補のフィルタリングがより厳密に行われ、マスクのエッジと重なる欠陥候補や指定サイズ範囲外の欠陥候補が除外されるよう
```

この修正版では以下の変更を行いました：

1. テンプレート画像のパスを正しく設定しました。
2. `template_matching` 関数内でテンプレート画像をグレースケールで読み込むようにしました。
3. `template_matching` 関数内で入力画像もグレースケールに変換するようにしました。
4. エラーハンドリングを追加し、テンプレート画像が見つからない場合などにユーザーフレンドリーなエラーメッセージを表示するようにしました。

注

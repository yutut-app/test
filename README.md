import streamlit as st
import cv2
import numpy as np
from skimage import io, filters, feature, measure
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from io import BytesIO
import PIL.Image
import os

# 1. ライブラリのインポート
# 必要なライブラリは上部でインポート済み

# 2. パラメータの設定
st.sidebar.title("画像処理パラメータ設定")

# ワーク接合部の削除
crop_width = st.sidebar.slider('ワーク接合部の削除: 幅 (px)', min_value=500, max_value=2000, value=1360)

# 二値化によるマスクの作成
st.sidebar.subheader("二値化によるマスクの作成")
threshold_value = st.sidebar.slider('二値化しきい値', min_value=0, max_value=255, value=150)
kernel_size = st.sidebar.slider('カーネルサイズ (奇数)', min_value=1, max_value=15, value=5, step=2)
iterations_open = st.sidebar.slider('膨張処理の繰り返し回数', min_value=1, max_value=10, value=3)
iterations_close = st.sidebar.slider('収縮処理の繰り返し回数', min_value=1, max_value=30, value=20)

# エッジ検出とテクスチャ検出
st.sidebar.subheader("エッジ検出とテクスチャ検出")
gaussian_kernel_size = st.sidebar.slider('ガウシアンブラーのカーネルサイズ (奇数)', min_value=3, max_value=15, value=7, step=2)
sigma = st.sidebar.slider('ガウシアンブラーの標準偏差', min_value=0.1, max_value=10.0, value=3.0)
canny_min_threshold = st.sidebar.slider('エッジ検出の最小しきい値', min_value=0, max_value=255, value=30)
canny_max_threshold = st.sidebar.slider('エッジ検出の最大しきい値', min_value=0, max_value=255, value=120)
texture_threshold = st.sidebar.slider('テクスチャの変化を検出するためのしきい値', min_value=0, max_value=50, value=15)

# エッジ補完
st.sidebar.subheader("エッジ補完")
edge_kernel_size = st.sidebar.slider('エッジ補完のカーネルサイズ', min_value=1, max_value=10, value=3)
edge_open_iterations = st.sidebar.slider('ノイズ削除の繰り返し回数', min_value=1, max_value=10, value=2)
edge_close_iterations = st.sidebar.slider('エッジ補完の繰り返し回数', min_value=1, max_value=10, value=2)

# マスクエッジ検出
st.sidebar.subheader("マスクエッジ検出")
mask_edge_min_threshold = st.sidebar.slider('マスクエッジ最小しきい値', min_value=0, max_value=255, value=100)
mask_edge_max_threshold = st.sidebar.slider('マスクエッジ最大しきい値', min_value=0, max_value=255, value=200)
mask_edge_margin = st.sidebar.slider('マスクエッジの余裕幅（ピクセル単位）', min_value=0, max_value=20, value=5)

# 欠陥サイズのフィルタリング
st.sidebar.subheader("欠陥サイズのフィルタリング")
min_defect_size = st.sidebar.slider('最小欠陥サイズ（ピクセル）', min_value=1, max_value=50, value=5)
max_defect_size = st.sidebar.slider('最大欠陥サイズ（ピクセル）', min_value=50, max_value=200, value=100)

# 欠陥候補の保存パラメータ
enlargement_factor = st.sidebar.slider('欠陥候補画像の拡大倍率', min_value=1, max_value=20, value=10)

# 3. 画像アップロード
st.title("欠陥検出アプリケーション")

st.header("画像アップロード")
uploaded_original_image = st.file_uploader("元画像をアップロード", type=['jpg', 'png'])
uploaded_keyence_image = st.file_uploader("キーエンス前処理画像をアップロード", type=['jpg', 'png'])

if uploaded_original_image is not None and uploaded_keyence_image is not None:
    original_image = PIL.Image.open(BytesIO(uploaded_original_image.read())).convert('RGB')
    keyence_image = PIL.Image.open(BytesIO(uploaded_keyence_image.read())).convert('RGB')
    
    # 画像表示
    st.subheader("アップロードされた画像")
    st.image([original_image, keyence_image], caption=["元画像", "キーエンス前処理画像"], use_column_width=True)
    
    # 4. ワーク接合部の削除
    def template_matching(image, template_path):
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc
    
    # テンプレートパスの設定（ユーザー環境に合わせてパスを調整してください）
    # ここではサンプルとして同じディレクトリ内にテンプレート画像があると仮定します
    right_template_path = "right_keyence.jpg"  # 実際のパスに置き換えてください
    left_template_path = "left_keyence.jpg"    # 実際のパスに置き換えてください
    
    # ワーク接合部の削除
    def remove_joint_part(original_img, keyence_img):
        right_val, _ = template_matching(keyence_img, right_template_path)
        left_val, _ = template_matching(keyence_img, left_template_path)
        
        if right_val > left_val:
            # ワークの右側：左からcrop_widthピクセルカット
            cropped_original = original_img[:, crop_width:]
            cropped_keyence = keyence_img[:, crop_width:]
        else:
            # ワークの左側：右からcrop_widthピクセルカット
            cropped_original = original_img[:, :-crop_width]
            cropped_keyence = keyence_img[:, :-crop_width]
        
        return cropped_original, cropped_keyence
    
    cropped_original, cropped_keyence = remove_joint_part(np.array(original_image), np.array(keyence_image))
    
    st.subheader("ワーク接合部削除後の画像")
    st.image([cropped_original, cropped_keyence], caption=["削除後元画像", "削除後キーエンス前処理画像"], use_column_width=True)
    
    # 5. 二値化によるマスクの作成
    def binarize_image(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
        return binary_image
    
    binary_mask = binarize_image(cropped_original)
    
    st.subheader("二値化によるマスク")
    st.image(binary_mask, caption="二値化マスク", use_column_width=True)
    
    # 6. エッジ検出とテクスチャ検出の改良
    def detect_edges_and_texture(cropped_keyence_image, binary_image):
        # H面のマスクを適用して背景を除去
        masked_image = cv2.bitwise_and(cropped_keyence_image, cropped_keyence_image, mask=binary_image)
        
        # ガウシアンブラーを適用してノイズを除去
        blurred_image = cv2.GaussianBlur(masked_image, (gaussian_kernel_size, gaussian_kernel_size), sigma)
        
        # エッジ検出（パラメータを低めに設定）
        edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
        
        # テクスチャの変化を検出（ラプラシアンを使用）
        laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
        abs_laplacian = np.absolute(laplacian)
        laplacian_edges = np.uint8(abs_laplacian > texture_threshold) * 255
        
        # エッジ検出とテクスチャ変化の結果を統合
        combined_edges = cv2.bitwise_or(edges, laplacian_edges)
        
        return combined_edges
    
    combined_edges = detect_edges_and_texture(cropped_keyence, binary_mask)
    
    st.subheader("エッジとテクスチャの検出結果")
    st.image(combined_edges, caption="エッジとテクスチャの検出結果", use_column_width=True)
    
    # 6. エッジの補完
    def complete_edges(edge_image, mask):
        # マスクエッジを検出
        mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
        
        # マスクエッジを膨張させ、余裕を持たせる
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * mask_edge_margin + 1, 2 * mask_edge_margin + 1))
        expanded_edges = cv2.dilate(mask_edges, kernel, iterations=1)
        
        # スケルトン化
        skeleton = skeletonize(edge_image > 0)
        skeleton = skeleton.astype(np.uint8) * 255
        
        # ノイズ削除
        kernel_morph = np.ones((edge_kernel_size, edge_kernel_size), np.uint8)
        opened_skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel_morph, iterations=edge_open_iterations)
        
        # エッジ補完
        connected_skeleton = cv2.morphologyEx(opened_skeleton, cv2.MORPH_CLOSE, kernel_morph, iterations=edge_close_iterations)
        
        # エッジの補完結果
        completed_edges = cv2.bitwise_or(edge_image, connected_skeleton)
        
        # マスクエッジと重なる部分を除外
        completed_edges = cv2.bitwise_and(completed_edges, cv2.bitwise_not(expanded_edges))
        
        return completed_edges
    
    completed_edges = complete_edges(combined_edges, binary_mask)
    
    st.subheader("エッジの補完結果")
    st.image(completed_edges, caption="エッジの補完結果", use_column_width=True)
    
    # 6.2 ラベリング処理と6.3 欠陥候補の中心座標の取得
    def label_and_filter_defects_with_margin(edge_image, mask, min_size, max_size, margin):
        # ラベリング処理
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge_image)
        
        # マスクのエッジを除外
        mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * margin + 1, 2 * margin + 1))
        expanded_mask_edges = cv2.dilate(mask_edges, kernel, iterations=1)
        
        # 欠陥候補のフィルタリング
        defect_candidates = []
        for i in range(1, num_labels):
            size = stats[i, cv2.CC_STAT_AREA]
            if min_size <= size <= max_size:
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]
                # 欠陥候補が拡張マスクエッジと重なっていないか確認
                if np.any(expanded_mask_edges[y:y+h, x:x+w] > 0):
                    continue  # エッジと重なっている場合は除外
                defect_candidates.append({'x': x, 'y': y, 'w': w, 'h': h, 'centroid_x': cx, 'centroid_y': cy, 'max_length': max(w, h)})
        
        return defect_candidates
    
    defects = label_and_filter_defects_with_margin(completed_edges, binary_mask, min_defect_size, max_defect_size, mask_edge_margin)
    
    # 7. 欠陥候補の表示
    def draw_defects(image, defects):
        result_image = image.copy()
        for defect in defects:
            x, y, w, h = defect['x'], defect['y'], defect['w'], defect['h']
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 青色の矩形
            cv2.circle(result_image, (int(defect['centroid_x']), int(defect['centroid_y'])), 3, (0, 0, 255), -1)  # 赤色の中心点
        return result_image
    
    result_defects_image = draw_defects(completed_edges, defects)
    
    st.subheader("欠陥候補の検出結果")
    st.image(result_defects_image, caption="欠陥候補の検出結果", use_column_width=True)
    
    # 8. 欠陥候補の保存（外接矩形を正方形で切り出し、100倍に拡大）
    def save_defect_candidates(defects, keyence_image, output_dir, image_name):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i, defect in enumerate(defects, 1):
            cx, cy = defect['centroid_x'], defect['centroid_y']
            max_length = int(defect['max_length'])
            half_side = max_length // 2
            x1 = max(int(cx) - half_side, 0)
            y1 = max(int(cy) - half_side, 0)
            x2 = min(int(cx) + half_side, keyence_image.shape[1])
            y2 = min(int(cy) + half_side, keyence_image.shape[0])
            
            # 正方形で切り出し
            defect_crop = keyence_image[y1:y2, x1:x2]
            
            # 100倍に拡大
            enlarged_defect = cv2.resize(defect_crop, (0, 0), fx=enlargement_factor, fy=enlargement_factor, interpolation=cv2.INTER_LINEAR)
            
            # 表示名の作成
            display_name = f"欠陥候補{i}(尺度{max_length}px)"
            
            # 欠陥候補画像を保存
            defect_filename = f"{display_name}.png"
            defect_filepath = os.path.join(output_dir, defect_filename)
            cv2.imwrite(defect_filepath, cv2.cvtColor(enlarged_defect, cv2.COLOR_RGB2BGR))
    
    # 保存先ディレクトリの設定
    output_defect_dir = os.path.join("output", "defect_candidate", os.path.splitext(uploaded_original_image.name)[0])
    save_defect_candidates(defects, cropped_keyence, output_defect_dir, uploaded_original_image.name)
    
    st.success(f"欠陥候補の画像を保存しました: {output_defect_dir}")
    
else:
    st.write("元画像とキーエンス前処理画像の両方をアップロードしてください。")

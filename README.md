はい、申し訳ありません。Cannyエッジ検出とDoGフィルタのパラメータも追加します。

```python
# 1. ライブラリのインポート
import os
import cv2
import numpy as np
from skimage import io, filters, feature, measure
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
```

```python
# 2. パラメータの設定

# ディレクトリとファイルパス
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"
processing_template_dir = os.path.join(input_data_dir, "processing_template")

# テンプレートファイルパス
right_processing_template_path = os.path.join(processing_template_dir, "right_processing.jpg")
left_processing_template_path = os.path.join(processing_template_dir, "left_processing.jpg")

# ラベル定義
ng_labels = ['label1', 'label2', 'label3']  # label1: 鋳巣, label2: 凹み, label3: 亀裂

# 円検出パラメータ
circle_dp = 1.0  # 分解能の逆比（1.0が標準、大きいほど検出感度が下がる）
circle_min_dist = 20  # 検出される円の中心間の最小距離（ピクセル）
circle_param1 = 50  # Cannyエッジ検出の高閾値（エッジ検出の感度）
circle_param2 = 30  # 円検出の閾値（小さいほど誤検出が増える）
circle_min_radius = 10  # 検出する円の最小半径（ピクセル）
circle_max_radius = 100  # 検出する円の最大半径（ピクセル）

# テンプレートマッチングパラメータ
template_match_threshold = 0.8  # マッチング判定の閾値（0-1、大きいほど厳密）

# スケール調整パラメータ
scale_min = 0.8  # 最小スケール倍率
scale_max = 1.2  # 最大スケール倍率
scale_step = 0.1  # スケール調整のステップ幅

# Cannyエッジ検出パラメータ（大きな欠陥用）
canny_kernel_size = (3, 3)  # ガウシアンフィルタのカーネルサイズ
canny_sigma = 2.0  # ガウシアンフィルタのシグマ値
canny_min_threshold = 55  # Cannyの最小閾値
canny_max_threshold = 250  # Cannyの最大閾値
canny_merge_distance = 15  # Canny検出結果の統合距離
texture_threshold = 30 # テクスチャ検出用閾値

# DoGフィルタパラメータ（小さな欠陥用）
dog_ksize = 9  # DoGフィルタのカーネルサイズ
dog_sigma1 = 1.5  # 1つ目のガウシアンフィルタのシグマ値
dog_sigma2 = 3.5  # 2つ目のガウシアンフィルタのシグマ値
dog_merge_distance = 15  # DoG検出結果の統合距離

# 動的閾値処理パラメータ
dynamic_ksize = 25  # 局所領域のサイズ
dynamic_c = 6  # 閾値調整用定数
dynamic_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C  # 適応的閾値処理の方法

# 欠陥サイズパラメータ
min_large_defect_size = 60  # 大きな欠陥の最小サイズ（ピクセル）
max_large_defect_size = 100  # 大きな欠陥の最大サイズ（ピクセル）
min_small_defect_size = 5   # 小さな欠陥の最小サイズ（ピクセル）
max_small_defect_size = 60  # 小さな欠陥の最大サイズ（ピクセル）

# エッジ補完パラメータ
edge_kernel_size = (3, 3)  # エッジ補完のカーネルサイズ
edge_open_iterations = 2   # ノイズ削除の繰り返し回数
edge_close_iterations = 10  # エッジ補完の繰り返し回数

# マスクエッジ検出パラメータ
mask_edge_min_threshold = 50  # マスクエッジ検出の最小閾値
mask_edge_max_threshold = 150  # マスクエッジ検出の最大閾値
mask_edge_margin = 10  # マスクエッジの余裕幅（ピクセル）

# コントラストと輝度パラメータ
min_contrast_ratio = 0.12  # 最小コントラスト比
min_intensity_diff = 25    # 最小輝度差
bright_threshold = 180     # 明るい領域の閾値
dark_threshold = 50        # 暗い領域の閾値

# ファイル出力パラメータ
enlargement_factor = 1  # 欠陥候補画像の保存時の拡大倍率
```

これらのパラメータは以下の目的で使用されます：
- 加工領域の検出と位置合わせ
- Cannyエッジ検出による大きな欠陥の検出
- DoGフィルタによる小さな欠陥の検出
- エッジの補完と欠陥候補の統合
- 欠陥候補のフィルタリング
- 結果の保存と出力

パラメータの値は画像の特性や検出したい欠陥の性質に応じて調整が必要になる場合があります。

続けて#3以降のコードを生成しますか？


はい、#3のコードを生成します。可視化も含めて記載します。

```python
# 3. データの読み込み

def load_origin_keyence_images(directory):
    """
    指定ディレクトリからNormalとShape画像のペアを読み込みます

    引数:
        directory (str): 画像が格納されているディレクトリのパス

    戻り値:
        list: (Normal画像パス, Shape画像パス, Shape画像ファイル名)のタプルのリスト
    """
    normal_images = {}
    shape_images = {}
    
    # 指定ディレクトリ内の全ファイルを走査
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "Normal" in file and file.endswith(".jpg"):
                base_name = file.replace("Normal", "")
                normal_images[base_name] = (os.path.join(root, file), file)
            elif "Shape" in file and file.endswith(".jpg"):
                base_name = file.replace("Shape", "")
                shape_images[base_name] = (os.path.join(root, file), file)
    
    # NormalとShapeの画像ペアを作成
    matched_images = []
    for base_name in normal_images:
        if base_name in shape_images:
            matched_images.append((normal_images[base_name][0], 
                                 shape_images[base_name][0], 
                                 shape_images[base_name][1]))
    
    return matched_images

def visualize_loaded_images(image_pairs, num_samples=1):
    """
    読み込んだ画像ペアを可視化します

    引数:
        image_pairs (list): 画像ペアのリスト
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(image_pairs))
    
    for i in range(num_samples):
        normal_path, shape_path, filename = image_pairs[i]
        
        # 画像を読み込み
        normal_img = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
        shape_img = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        
        # 表示
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(normal_img, cmap='gray')
        axes[0].set_title('Normal Image')
        axes[0].axis('off')
        
        axes[1].imshow(shape_img, cmap='gray')
        axes[1].set_title('Shape Image')
        axes[1].axis('off')
        
        plt.suptitle(f'Image Pair: {filename}')
        plt.tight_layout()
        plt.show()

# NG画像とOK画像をそれぞれ読み込む
ng_images_label1 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label1"))
#ng_images_label2 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label2"))
#ng_images_label3 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", "label3"))
#ok_images = load_origin_keyence_images(os.path.join(input_data_dir, "OK"))

# 読み込んだ画像を可視化（各カテゴリから1サンプルずつ）
print("Visualizing NG Label 1 images:")
visualize_loaded_images(ng_images_label1, num_samples=1)
#print("\nVisualizing NG Label 2 images:")
#visualize_loaded_images(ng_images_label2, num_samples=1)
#print("\nVisualizing NG Label 3 images:")
#visualize_loaded_images(ng_images_label3, num_samples=1)
#print("\nVisualizing OK images:")
#visualize_loaded_images(ok_images, num_samples=1)

# 読み込んだ画像数を表示
print(f"\nLoaded images count:")
print(f"NG Label 1: {len(ng_images_label1)} pairs")
#print(f"NG Label 2: {len(ng_images_label2)} pairs")
#print(f"NG Label 3: {len(ng_images_label3)} pairs")
#print(f"OK: {len(ok_images)} pairs")
```

このコードの特徴：
1. 明確な関数のドキュメント文字列を含む
2. 可視化機能を組み込み
3. コードの各部分が単一の責任を持つ
4. エラーハンドリングは次のバージョンで追加可能
5. メモリ使用を考慮してサンプル数を制限可能

続けて#4以降のコードを生成しますか？

はい、画像処理と画像表示を分離し、各項目ごとに実行できるように修正します。まず、#4のコードから開始します。

```python
# 4. テンプレートマッチングによる加工領域の検出

def detect_circles(image):
    """
    画像から複数の円を検出します

    引数:
        image (numpy.ndarray): 入力画像（グレースケール）

    戻り値:
        numpy.ndarray or None: 検出された円の情報（中心座標と半径）。検出失敗時はNone
    """
    circles = cv2.HoughCircles(image, 
                              cv2.HOUGH_GRADIENT, 
                              dp=circle_dp,
                              minDist=circle_min_dist,
                              param1=circle_param1,
                              param2=circle_param2,
                              minRadius=circle_min_radius,
                              maxRadius=circle_max_radius)
    
    if circles is not None:
        return np.uint16(np.around(circles[0]))
    return None

def get_optimal_scale_and_transform(template_circles, target_circles):
    """
    複数の円の中心点から最適なスケールと変換行列を計算します

    引数:
        template_circles (numpy.ndarray): テンプレート画像の円の情報
        target_circles (numpy.ndarray): 対象画像の円の情報

    戻り値:
        tuple: (最適なスケール値, 変換行列)。失敗時は(None, None)
    """
    if len(template_circles) < 3 or len(target_circles) < 3:
        return None, None
    
    best_scale = 1.0
    min_error = float('inf')
    best_matrix = None
    
    for scale in np.arange(scale_min, scale_max + scale_step, scale_step):
        scaled_template_pts = template_circles[:3, :2].astype(np.float32) * scale
        target_pts = target_circles[:3, :2].astype(np.float32)
        
        M = cv2.getAffineTransform(scaled_template_pts, target_pts)
        transformed_pts = cv2.transform(scaled_template_pts.reshape(1, -1, 2), M)
        error = np.sum(np.sqrt(np.sum((transformed_pts - target_pts) ** 2, axis=2)))
        
        if error < min_error:
            min_error = error
            best_scale = scale
            best_matrix = M
    
    return best_scale, best_matrix

def create_processing_area_mask(template_path, target_image):
    """
    テンプレート画像から加工領域のマスクを作成します

    引数:
        template_path (str): テンプレート画像のパス
        target_image (numpy.ndarray): 対象画像

    戻り値:
        numpy.ndarray or None: 作成されたマスク画像。失敗時はNone
    """
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    inverted_template = cv2.bitwise_not(template)
    
    template_circles = detect_circles(template)
    target_circles = detect_circles(target_image)
    
    if template_circles is None or target_circles is None:
        return None
    
    scale, M = get_optimal_scale_and_transform(template_circles, target_circles)
    if M is None:
        return None
    
    scaled_template = cv2.resize(inverted_template, None, fx=scale, fy=scale)
    rows, cols = target_image.shape
    aligned_template = cv2.warpAffine(scaled_template, M, (cols, rows))
    
    return aligned_template

def process_image_with_template(image_pair):
    """
    画像ペアに対してテンプレートマッチングと加工領域の検出を行います

    引数:
        image_pair (tuple): (Normal画像パス, Shape画像パス, ファイル名)のタプル

    戻り値:
        tuple: (処理済み画像, 元の画像, ファイル名)のタプル
    """
    origin_image_path, keyence_image_path, original_filename = image_pair
    image = cv2.imread(origin_image_path, cv2.IMREAD_GRAYSCALE)
    keyence_image = cv2.imread(keyence_image_path, cv2.IMREAD_GRAYSCALE)
    
    # 左右判定
    right_res = cv2.matchTemplate(keyence_image, 
                                 cv2.imread(right_processing_template_path, cv2.IMREAD_GRAYSCALE),
                                 cv2.TM_CCOEFF_NORMED)
    left_res = cv2.matchTemplate(keyence_image, 
                                cv2.imread(left_processing_template_path, cv2.IMREAD_GRAYSCALE),
                                cv2.TM_CCOEFF_NORMED)
    
    template_path = right_processing_template_path if np.max(right_res) > np.max(left_res) else left_processing_template_path
    
    # マスク作成と適用
    mask = create_processing_area_mask(template_path, keyence_image)
    if mask is None:
        return image, keyence_image, original_filename
    
    processed_image = cv2.bitwise_and(image, image, mask=mask)
    processed_keyence = cv2.bitwise_and(keyence_image, keyence_image, mask=mask)
    
    return processed_image, processed_keyence, original_filename

# 画像の処理を実行
processed_ng_images_label1 = [process_image_with_template(pair) for pair in ng_images_label1]
#processed_ng_images_label2 = [process_image_with_template(pair) for pair in ng_images_label2]
#processed_ng_images_label3 = [process_image_with_template(pair) for pair in ng_images_label3]
#processed_ok_images = [process_image_with_template(pair) for pair in ok_images]
```

続けて、可視化用のコードブロックを記載します。次のメッセージに続きます。

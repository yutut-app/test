承知しました。まず、検出機能をより小さな責任に分割し、処理をモジュール化して可読性を向上させます。#5から始めます。

```python
# 5. Canny+DoGによる欠陥検出

def apply_gaussian_and_canny(image, kernel_size, sigma, min_threshold, max_threshold):
    """
    画像にガウシアンフィルタとCannyエッジ検出を適用します

    引数：
        image (numpy.ndarray): 入力画像
        kernel_size (tuple): ガウシアンフィルタのカーネルサイズ
        sigma (float): ガウシアンフィルタのシグマ値
        min_threshold (int): Cannyエッジ検出の最小閾値
        max_threshold (int): Cannyエッジ検出の最大閾値

    戻り値：
        numpy.ndarray: エッジ検出結果
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    return cv2.Canny(blurred, min_threshold, max_threshold)

def detect_texture(image, threshold):
    """
    テクスチャを検出します

    引数：
        image (numpy.ndarray): 入力画像
        threshold (float): テクスチャ検出の閾値

    戻り値：
        numpy.ndarray: テクスチャ検出結果
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    return np.uint8(abs_laplacian > threshold) * 255

def merge_regions(image, kernel_size, operation=cv2.MORPH_CLOSE):
    """
    近接領域を統合します

    引数：
        image (numpy.ndarray): 入力画像
        kernel_size (int): 統合に使用するカーネルサイズ
        operation (int): モルフォロジー演算の種類

    戻り値：
        numpy.ndarray: 統合後の画像
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, operation, kernel)

def detect_edges_and_texture(cropped_keyence_image, binarized_image):
    """
    Cannyエッジ検出とテクスチャ検出による大きな鋳巣の検出を行います

    引数：
        cropped_keyence_image (numpy.ndarray): 入力画像
        binarized_image (numpy.ndarray): マスク画像

    戻り値：
        numpy.ndarray: 検出結果
    """
    # マスク適用
    masked_image = cv2.bitwise_and(cropped_keyence_image, cropped_keyence_image, 
                                  mask=binarized_image)
    
    # エッジ検出
    edges = apply_gaussian_and_canny(masked_image, canny_kernel_size, canny_sigma,
                                   canny_min_threshold, canny_max_threshold)
    
    # テクスチャ検出
    texture = detect_texture(masked_image, texture_threshold)
    
    # 結果の統合
    combined = cv2.bitwise_or(edges, texture)
    
    # 近接領域の統合
    return merge_regions(combined, canny_merge_distance)
```

続きは次のメッセージで記載します。

はい、DoG検出部分の改良コードを記載します。

```python
def calculate_dog(image, ksize, sigma1, sigma2):
    """
    Difference of Gaussianを計算します

    引数：
        image (numpy.ndarray): 入力画像
        ksize (int): カーネルサイズ
        sigma1 (float): 1つ目のガウシアンのシグマ値
        sigma2 (float): 2つ目のガウシアンのシグマ値

    戻り値：
        numpy.ndarray: DoG結果
    """
    g1 = cv2.GaussianBlur(image, (ksize, ksize), sigma1)
    g2 = cv2.GaussianBlur(image, (ksize, ksize), sigma2)
    return g1 - g2

def calculate_gradient_magnitude(image):
    """
    画像の勾配強度を計算します

    引数：
        image (numpy.ndarray): 入力画像

    戻り値：
        numpy.ndarray: 正規化された勾配強度
    """
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def create_intensity_masks(image, bright_thresh, dark_thresh):
    """
    明るい領域と暗い領域のマスクを作成します

    引数：
        image (numpy.ndarray): 入力画像
        bright_thresh (int): 明るい領域の閾値
        dark_thresh (int): 暗い領域の閾値

    戻り値：
        tuple: (明るい領域のマスク, 暗い領域のマスク)
    """
    _, bright_mask = cv2.threshold(image, bright_thresh, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(image, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    return bright_mask, dark_mask

def detect_defects_dog_dynamic(cropped_keyence_image, binarized_image):
    """
    DoGフィルタと動的閾値処理による小さな鋳巣の検出を行います

    引数：
        cropped_keyence_image (numpy.ndarray): 入力画像
        binarized_image (numpy.ndarray): マスク画像

    戻り値：
        numpy.ndarray: 検出結果
    """
    # 明暗領域の検出
    bright_mask, dark_mask = create_intensity_masks(cropped_keyence_image, 
                                                  bright_threshold, dark_threshold)
    
    # マルチスケールDoGの適用
    dog_results = []
    sigma_pairs = [(1.5, 3.5), (2.0, 4.0), (1.0, 2.5)]
    
    for sigma1, sigma2 in sigma_pairs:
        dog_result = calculate_dog(cropped_keyence_image, dog_ksize, sigma1, sigma2)
        normalized_dog = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dog_results.append(normalized_dog)
    
    combined_dog = np.maximum.reduce(dog_results)
    
    # 動的閾値処理
    binary_dog = cv2.adaptiveThreshold(cropped_keyence_image, 255, dynamic_method, 
                                     cv2.THRESH_BINARY_INV, dynamic_ksize, dynamic_c)
    
    # 勾配とコントラストの計算
    gradient_magnitude = calculate_gradient_magnitude(cropped_keyence_image)
    local_mean = cv2.blur(cropped_keyence_image, (dynamic_ksize, dynamic_ksize))
    intensity_diff = cv2.absdiff(cropped_keyence_image, local_mean)
    
    # コントラストマスクの作成
    local_std = np.std(cropped_keyence_image)
    contrast_ratio = intensity_diff / (local_std + 1e-6)
    contrast_mask = (contrast_ratio > min_contrast_ratio).astype(np.uint8) * 255
    
    # 結果の統合
    bright_result = cv2.bitwise_and(bright_mask, gradient_magnitude)
    dark_result = cv2.bitwise_and(dark_mask, gradient_magnitude)
    combined_mask = cv2.bitwise_or(bright_result, dark_result)
    
    result = cv2.bitwise_and(combined_mask, combined_dog)
    result = cv2.bitwise_and(result, contrast_mask)
    result = cv2.bitwise_and(result, binary_dog)
    result = cv2.bitwise_and(result, result, mask=binarized_image)
    
    # 近接領域の統合
    merged_result = merge_regions(result, dog_merge_distance)
    
    return cv2.bitwise_or(merged_result, cv2.bitwise_and(bright_mask, binarized_image))
```

続いてメイン処理部分を記載します。次のメッセージに続きます。

はい、メイン処理部分と可視化部分を記載します。

```python
def combine_defect_detection(image, mask):
    """
    CannyとDoGの検出結果を組み合わせて欠陥検出を行います

    引数：
        image (numpy.ndarray): 入力画像
        mask (numpy.ndarray): マスク画像

    戻り値：
        tuple: (統合結果, 大きな欠陥の検出結果, 小さな欠陥の検出結果)
    """
    # 大きな鋳巣の検出（Canny）
    large_defects = detect_edges_and_texture(image, mask)
    
    # 小さな鋳巣の検出（DoG）
    small_defects = detect_defects_dog_dynamic(image, mask)
    
    # 結果の統合
    combined_result = cv2.bitwise_or(large_defects, small_defects)
    
    return combined_result, large_defects, small_defects

def process_images_for_defect_detection(processed_images):
    """
    全画像に対して欠陥検出を実行します

    引数：
        processed_images (list): 処理対象の画像リスト

    戻り値：
        list: 処理結果のリスト
    """
    detected_images = []
    for binarized_image, cropped_keyence_image, original_filename in processed_images:
        combined_result, large_defects, small_defects = combine_defect_detection(
            cropped_keyence_image, binarized_image)
        detected_images.append((binarized_image, combined_result, large_defects, 
                              small_defects, original_filename))
    return detected_images

# NGとOK画像に対して欠陥検出を実行
detected_ng_images_label1 = process_images_for_defect_detection(processed_ng_images_label1)
#detected_ng_images_label2 = process_images_for_defect_detection(processed_ng_images_label2)
#detected_ng_images_label3 = process_images_for_defect_detection(processed_ng_images_label3)
#detected_ok_images = process_images_for_defect_detection(processed_ok_images)
```

```python
# 検出結果の可視化
def visualize_defect_detection(detected_images, num_samples=1):
    """
    欠陥検出結果を可視化します

    引数：
        detected_images (list): 検出結果のリスト
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(detected_images))
    
    for i in range(num_samples):
        binarized_image, combined_result, large_defects, small_defects, filename = detected_images[i]
        
        # 2x2のサブプロットを作成
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # オリジナル画像
        axes[0,0].imshow(binarized_image, cmap='gray')
        axes[0,0].set_title('Original Image with Mask')
        axes[0,0].axis('off')
        
        # Canny検出結果
        axes[0,1].imshow(large_defects, cmap='gray')
        axes[0,1].set_title('Large Defects (Canny)')
        axes[0,1].axis('off')
        
        # DoG検出結果
        axes[1,0].imshow(small_defects, cmap='gray')
        axes[1,0].set_title('Small Defects (DoG)')
        axes[1,0].axis('off')
        
        # 統合結果
        axes[1,1].imshow(combined_result, cmap='gray')
        axes[1,1].set_title('Combined Detection Results')
        axes[1,1].axis('off')
        
        plt.suptitle(f'Defect Detection Results - {filename}', y=1.02)
        plt.tight_layout()
        plt.show()

# 検出結果の可視化
print("Visualizing detection results for NG Label 1:")
visualize_defect_detection(detected_ng_images_label1, num_samples=1)

# 処理した画像数を表示
print(f"\nProcessed images count:")
print(f"NG Label 1: {len(detected_ng_images_label1)} images")
#print(f"NG Label 2: {len(detected_ng_images_label2)} images")
#print(f"NG Label 3: {len(detected_ng_images_label3)} images")
#print(f"OK: {len(detected_ok_images)} images")
```

主な改良点：
1. 各関数の責任を明確に分離
2. 共通処理を独立した関数として抽出
3. 詳細なドキュメント文字列の追加
4. 処理と可視化の分離
5. 中間結果の確認が容易な可視化機能

続けて#6以降のコードを生成しますか？

はい、エッジ補完とフィルタリング処理を改良したコードを生成します。

```python
# 7. エッジの補完とラベリング処理

def create_mask_edge_margin(mask, margin):
    """
    マスクのエッジ領域に余裕を持たせたマージンを作成します

    引数：
        mask (numpy.ndarray): 入力マスク画像
        margin (int): マージンのピクセル幅

    戻り値：
        numpy.ndarray: マージン付きのエッジマスク
    """
    mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
    return cv2.dilate(mask_edges, kernel, iterations=1)

def apply_morphology(image, kernel, operation, iterations):
    """
    モルフォロジー演算を適用します

    引数：
        image (numpy.ndarray): 入力画像
        kernel (numpy.ndarray): カーネル
        operation: モルフォロジー演算の種類
        iterations (int): 繰り返し回数

    戻り値：
        numpy.ndarray: 処理結果
    """
    return cv2.morphologyEx(image, operation, kernel, iterations=iterations)

def complete_edges(edge_image, mask):
    """
    エッジを補完して連続性を向上させます

    引数：
        edge_image (numpy.ndarray): エッジ画像
        mask (numpy.ndarray): マスク画像

    戻り値：
        numpy.ndarray: 補完されたエッジ画像
    """
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    skeleton = skeletonize(edge_image > 0)
    kernel = np.ones(edge_kernel_size, np.uint8)
    
    # ノイズ除去と接続処理
    skeleton_processed = skeleton.astype(np.uint8)
    skeleton_processed = apply_morphology(skeleton_processed, kernel, cv2.MORPH_OPEN, edge_open_iterations)
    skeleton_processed = apply_morphology(skeleton_processed, kernel, cv2.MORPH_CLOSE, edge_close_iterations)
    
    # 結果の統合
    completed_edges = np.maximum(edge_image, skeleton_processed * 255)
    return np.where(mask_edges_with_margin > 0, edge_image, completed_edges).astype(np.uint8)

def process_images_for_edge_completion(detected_images):
    """
    全画像に対してエッジ補完を実行します

    引数：
        detected_images (list): 検出済み画像のリスト

    戻り値：
        list: エッジ補完済み画像のリスト
    """
    completed_edge_images = []
    for binarized_image, edge_image, original_filename in detected_images:
        completed_edges = complete_edges(edge_image, binarized_image)
        completed_edge_images.append((binarized_image, completed_edges, original_filename))
    return completed_edge_images

# NGとOK画像に対してエッジ補完を実行
completed_ng_images_label1 = process_images_for_edge_completion(detected_ng_images_label1)
#completed_ng_images_label2 = process_images_for_edge_completion(detected_ng_images_label2)
#completed_ng_images_label3 = process_images_for_edge_completion(detected_ng_images_label3)
#completed_ok_images = process_images_for_edge_completion(detected_ok_images)
```

```python
# エッジ補完結果の可視化
def visualize_edge_completion(completed_images, num_samples=1):
    """
    エッジ補完結果を可視化します

    引数：
        completed_images (list): エッジ補完済み画像のリスト
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(completed_images))
    
    for i in range(num_samples):
        binarized_image, completed_edges, filename = completed_images[i]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # マスク画像
        axes[0].imshow(binarized_image, cmap='gray')
        axes[0].set_title('Binary Mask')
        axes[0].axis('off')
        
        # 補完済みエッジ
        axes[1].imshow(completed_edges, cmap='gray')
        axes[1].set_title('Completed Edges')
        axes[1].axis('off')
        
        plt.suptitle(f'Edge Completion Results - {filename}')
        plt.tight_layout()
        plt.show()

# エッジ補完結果の可視化
print("Visualizing edge completion results for NG Label 1:")
visualize_edge_completion(completed_ng_images_label1, num_samples=1)
```

フィルタリング処理のコードは次のメッセージで続けます。

はい、フィルタリング処理のコードを記載します。

```python
# 8. 欠陥候補のフィルタリング

def extract_defect_region_info(region, detection_method):
    """
    領域から欠陥情報を抽出します

    引数：
        region: regionpropsで得られた領域情報
        detection_method (str): 検出方法の識別子

    戻り値：
        dict: 欠陥情報を含む辞書
    """
    y, x = region.bbox[0], region.bbox[1]
    h, w = region.bbox[2] - y, region.bbox[3] - x
    
    return {
        'label': region.label,
        'x': x, 'y': y, 'width': w, 'height': h,
        'area': region.area,
        'centroid_y': region.centroid[0],
        'centroid_x': region.centroid[1],
        'perimeter': region.perimeter,
        'eccentricity': region.eccentricity,
        'orientation': region.orientation,
        'major_axis_length': region.major_axis_length,
        'minor_axis_length': region.minor_axis_length,
        'solidity': region.solidity,
        'extent': region.extent,
        'aspect_ratio': max(w, h) / min(w, h) if min(w, h) > 0 else 0,
        'max_length': max(w, h),
        'detection_method': detection_method
    }

def filter_and_measure_defects(edge_image, mask, min_size, max_size):
    """
    エッジ画像から欠陥候補を検出し、特徴量を計測します

    引数：
        edge_image (numpy.ndarray): エッジ画像
        mask (numpy.ndarray): マスク画像
        min_size (int): 最小欠陥サイズ
        max_size (int): 最大欠陥サイズ

    戻り値：
        list: 欠陥情報のリスト
    """
    # マスクエッジの除外処理
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    binary_edge_image = (edge_image > 0).astype(np.uint8)
    binary_edge_image[mask_edges_with_margin > 0] = 0
    
    # ラベリングと特徴量計測
    labels = measure.label(binary_edge_image, connectivity=2)
    defects = []
    
    for region in measure.regionprops(labels):
        if min_size <= region.area <= max_size:
            detection_method = 'canny' if region.area >= min_large_defect_size else 'dog'
            defect_info = extract_defect_region_info(region, detection_method)
            defects.append(defect_info)
    
    return defects

def process_images_for_filtering(completed_edge_images):
    """
    全画像に対して欠陥候補のフィルタリングを実行します

    引数：
        completed_edge_images (list): エッジ補完済み画像のリスト

    戻り値：
        list: フィルタリング結果のリスト
    """
    filtered_images = []
    for binarized_image, edge_image, original_filename in completed_edge_images:
        defects = filter_and_measure_defects(
            edge_image, binarized_image,
            min_small_defect_size, max_large_defect_size
        )
        filtered_images.append((original_filename, binarized_image, edge_image, defects))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images_label1 = process_images_for_filtering(completed_ng_images_label1)
#filtered_ng_images_label2 = process_images_for_filtering(completed_ng_images_label2)
#filtered_ng_images_label3 = process_images_for_filtering(completed_ng_images_label3)
#filtered_ok_images = process_images_for_filtering(completed_ok_images)
```

```python
# フィルタリング結果の可視化
def visualize_filtered_defects(image_name, image, defects, mask):
    """
    フィルタリング結果を可視化します

    引数：
        image_name (str): 画像の名前
        image (numpy.ndarray): 入力画像
        defects (list): 検出された欠陥のリスト
        mask (numpy.ndarray): マスク画像
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image, cmap='gray')
    
    # マスクエッジの可視化
    mask_edges = create_mask_edge_margin(mask, mask_edge_margin)
    ax.imshow(mask_edges, alpha=0.3, cmap='cool')
    
    # 凡例の作成
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor='red', label='Large defect (Canny)'),
        patches.Patch(facecolor='none', edgecolor='blue', label='Small defect (DoG)')
    ]
    
    # 欠陥の可視化
    for defect in defects:
        color = 'red' if defect['detection_method'] == 'canny' else 'blue'
        rect = plt.Rectangle(
            (defect['x'], defect['y']),
            defect['width'], defect['height'],
            fill=False, edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(defect['x'], defect['y'], str(defect['label']), 
                color=color, fontsize=12)
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.title(f"Filtered Defects with Mask Edges - {image_name}\n"
              f"Red: Canny (Large), Blue: DoG (Small)", fontsize=20)
    plt.axis('off')
    plt.show()

# フィルタリング結果の可視化（例：最初のNG画像）
if filtered_ng_images_label1:
    image_name, binarized_image, edge_image, filtered_defects = filtered_ng_images_label1[0]
    visualize_filtered_defects(image_name, edge_image, filtered_defects, binarized_image)

# 処理結果の統計を表示
print("\nDetected defects summary:")
if filtered_ng_images_label1:
    total_defects = sum(len(item[3]) for item in filtered_ng_images_label1)
    print(f"NG Label 1: {total_defects} defects in {len(filtered_ng_images_label1)} images")
```

改良点：
1. 関数の責任を明確に分割（特に特徴量抽出部分）
2. 詳細なドキュメント文字列の追加
3. 変数名の明確化
4. 可視化機能の強化
5. 処理結果の統計情報の追加

続けて#9のコードを生成しますか？

はい、欠陥候補の画像保存とCSV出力の部分を改良します。

```python
# 9. 欠陥候補の画像の保存とCSV出力

def calculate_defect_region(defect, image_shape, margin=1):
    """
    欠陥領域の座標を計算します

    引数：
        defect (dict): 欠陥情報
        image_shape (tuple): 画像のサイズ
        margin (int): 余白のピクセル数

    戻り値：
        tuple: (y1, y2, x1, x2) の形式で切り出し範囲を返す
    """
    x, y = defect['x'], defect['y']
    width, height = defect['width'], defect['height']
    
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + width + margin, image_shape[1])
    y2 = min(y + height + margin, image_shape[0])
    
    return y1, y2, x1, x2

def save_defect_region(image, region_coords, output_path, enlargement_factor):
    """
    欠陥領域を切り出して保存します

    引数：
        image (numpy.ndarray): 入力画像
        region_coords (tuple): 切り出し範囲の座標
        output_path (str): 保存先のパス
        enlargement_factor (float): 拡大倍率

    戻り値：
        str: 保存したファイルの名前
    """
    y1, y2, x1, x2 = region_coords
    defect_region = image[y1:y2, x1:x2]
    enlarged_region = cv2.resize(defect_region, (0, 0), 
                               fx=enlargement_factor, fy=enlargement_factor)
    cv2.imwrite(output_path, enlarged_region)
    return os.path.basename(output_path)

def create_defect_data(defect, image_name, edge_filename, original_filename, image_type, image_label):
    """
    欠陥データの辞書を作成します

    引数：
        defect (dict): 欠陥情報
        image_name (str): 元画像の名前
        edge_filename (str): エッジ画像のファイル名
        original_filename (str): オリジナル画像のファイル名
        image_type (str): 画像タイプ（"ng" or "ok"）
        image_label (int): 画像ラベル

    戻り値：
        dict: 欠陥データの辞書
    """
    base_name = image_name.replace(".jpg", "")
    return {
        'image_name': image_name,
        'defect_image_name': f"defect_{defect['label']}.png",
        'defect_image_edge_path': os.path.join(image_type, base_name, "edge", edge_filename),
        'defect_image_original_path': os.path.join(image_type, base_name, "original", original_filename),
        'Image_label': image_label,
        'defect_label': 0,
        'detection_method': defect['detection_method'],
        **defect
    }

def save_defect_image(edge_image, original_image, defect, output_dir_edge, 
                     output_dir_original, image_name, defect_number):
    """
    欠陥候補領域の画像を保存します

    引数：
        edge_image (numpy.ndarray): エッジ画像
        original_image (numpy.ndarray): オリジナル画像
        defect (dict): 欠陥情報
        output_dir_edge (str): エッジ画像の出力ディレクトリ
        output_dir_original (str): オリジナル画像の出力ディレクトリ
        image_name (str): 画像名
        defect_number (int): 欠陥番号

    戻り値：
        tuple: (エッジ画像ファイル名, オリジナル画像ファイル名)
    """
    # 切り出し範囲の計算
    region_coords = calculate_defect_region(defect, edge_image.shape)
    
    # ファイル名の生成
    filename = f"defect_{defect_number}.png"
    edge_path = os.path.join(output_dir_edge, filename)
    original_path = os.path.join(output_dir_original, filename)
    
    # 画像の保存
    edge_filename = save_defect_region(edge_image, region_coords, edge_path, enlargement_factor)
    original_filename = save_defect_region(original_image, region_coords, original_path, enlargement_factor)
    
    return edge_filename, original_filename

def get_original_image(binarized_images, target_filename):
    """
    ファイル名に対応するオリジナル画像を取得します

    引数：
        binarized_images (list): 二値化済み画像のリスト
        target_filename (str): 検索するファイル名

    戻り値：
        numpy.ndarray or None: 対応するオリジナル画像、見つからない場合はNone
    """
    for binarized_image, cropped_keyence_image, original_filename in binarized_images:
        if original_filename == target_filename:
            return cropped_keyence_image
    return None

```

続きは次のメッセージで送ります。

はい、続きのコードを記載します。

```python
def process_images_for_saving(filtered_images, binarized_images, base_output_dir, image_label):
    """
    フィルタリングされた画像を保存し、欠陥データを生成します

    引数：
        filtered_images (list): フィルタリング済み画像のリスト
        binarized_images (list): 二値化済み画像のリスト
        base_output_dir (str): 出力先のベースディレクトリ
        image_label (int): 画像ラベル（0: OK, 1: NG）

    戻り値：
        list: 欠陥データのリスト
    """
    if filtered_images is None:
        return []
    
    defects_data = []
    image_type = "ng" if image_label == 1 else "ok"
    
    for original_filename, binarized_image, edge_image, defects in filtered_images:
        # ディレクトリの作成
        base_dir = os.path.join(base_output_dir, image_type, 
                               original_filename.replace(".jpg", ""))
        output_dir_edge = os.path.join(base_dir, "edge")
        output_dir_original = os.path.join(base_dir, "original")
        os.makedirs(output_dir_edge, exist_ok=True)
        os.makedirs(output_dir_original, exist_ok=True)
        
        # オリジナル画像の取得
        original_image = get_original_image(binarized_images, original_filename)
        if original_image is None:
            continue
        
        # 各欠陥の処理
        for defect in defects:
            edge_filename, original_filename_cut = save_defect_image(
                edge_image, original_image, defect,
                output_dir_edge, output_dir_original,
                original_filename, defect['label']
            )
            
            defect_data = create_defect_data(
                defect, original_filename, edge_filename,
                original_filename_cut, image_type, image_label
            )
            defects_data.append(defect_data)
    
    return defects_data

def save_to_csv(defects_data, output_dir, filename="defects_data.csv"):
    """
    欠陥データをCSVファイルに保存します

    引数：
        defects_data (list): 欠陥データのリスト
        output_dir (str): 出力ディレクトリ
        filename (str): 出力ファイル名
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df = pd.DataFrame(defects_data)
    
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
        print(f"Appended defects data to existing file: {output_path}")
    else:
        df.to_csv(output_path, index=False)
        print(f"Created new defects data file: {output_path}")
    
    print(f"Total defects saved: {len(defects_data)}")

# メインの処理実行
output_dir = os.path.join(output_data_dir, "defect_images")
os.makedirs(output_dir, exist_ok=True)

# 欠陥データの収集
all_defects_data = []
image_pairs = [
    ("NG Label 1", filtered_ng_images_label1, binarized_ng_images_label1, 1),
    #("NG Label 2", filtered_ng_images_label2, binarized_ng_images_label2, 1),
    #("NG Label 3", filtered_ng_images_label3, binarized_ng_images_label3, 1),
    #("OK", filtered_ok_images, binarized_ok_images, 0)
]

for name, filtered_images, binarized_images, label in image_pairs:
    if filtered_images:
        print(f"Processing {name} images...")
        defects_data = process_images_for_saving(filtered_images, binarized_images, 
                                               output_dir, label)
        all_defects_data.extend(defects_data)
        print(f"Processed {len(defects_data)} defects from {name}")
    else:
        print(f"No data for {name} images.")

# CSVファイルへの保存
csv_output_dir = os.path.join(output_data_dir, "defect_data")
save_to_csv(all_defects_data, csv_output_dir)
```

```python
# 保存結果の可視化例
def visualize_saved_defects(defects_data, num_samples=1):
    """
    保存された欠陥画像をサンプル表示します

    引数：
        defects_data (list): 欠陥データのリスト
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(defects_data))
    selected_samples = defects_data[:num_samples]
    
    for data in selected_samples:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        edge_image = cv2.imread(os.path.join(output_data_dir, data['defect_image_edge_path']))
        original_image = cv2.imread(os.path.join(output_data_dir, data['defect_image_original_path']))
        
        axes[0].imshow(cv2.cvtColor(edge_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Edge Image')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Original Image')
        axes[1].axis('off')
        
        plt.suptitle(f"Defect {data['label']} from {data['image_name']}\n"
                    f"Detection Method: {data['detection_method']}")
        plt.tight_layout()
        plt.show()

# 保存結果の可視化
if all_defects_data:
    print("\nVisualizing saved defect samples:")
    visualize_saved_defects(all_defects_data, num_samples=1)
```

改良点：
1. 各関数の責任を明確に分離
2. 詳細なドキュメント文字列の追加
3. 画像処理の共通部分を関数化
4. エラーハンドリングの改善
5. 進捗報告の強化
6. 可視化機能の追加

これで、すべての処理（#1から#9まで）のコードが完成しました。

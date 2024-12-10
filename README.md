可視化用のコードブロックを記載します。

```python
# 4. テンプレートマッチングによる加工領域の検出結果の可視化

def visualize_processing_results(processed_images, num_samples=1):
    """
    加工領域検出の結果を可視化します

    引数:
        processed_images (list): 処理済み画像のリスト
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(processed_images))
    
    for i in range(num_samples):
        processed_image, processed_keyence, filename = processed_images[i]
        
        # 3x2のサブプロットを作成
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 元画像（Normal）
        axes[0,0].imshow(processed_image, cmap='gray')
        axes[0,0].set_title('Processed Normal Image')
        axes[0,0].axis('off')
        
        # 処理後のShape画像
        axes[0,1].imshow(processed_keyence, cmap='gray')
        axes[0,1].set_title('Processed Shape Image')
        axes[0,1].axis('off')
        
        # 元画像と処理後の差分（Normal）
        if 'origin_image_path' in locals():
            origin_image = cv2.imread(origin_image_path, cv2.IMREAD_GRAYSCALE)
            diff_normal = cv2.absdiff(origin_image, processed_image)
            axes[1,0].imshow(diff_normal, cmap='gray')
            axes[1,0].set_title('Difference (Normal)')
            axes[1,0].axis('off')
        
        # 処理結果の強調表示
        highlighted = cv2.addWeighted(
            processed_keyence, 0.7,
            cv2.threshold(processed_keyence, 127, 255, cv2.THRESH_BINARY)[1], 0.3,
            0
        )
        axes[1,1].imshow(highlighted, cmap='gray')
        axes[1,1].set_title('Highlighted Processing Area')
        axes[1,1].axis('off')
        
        plt.suptitle(f'Processing Results - {filename}', y=1.02)
        plt.tight_layout()
        plt.show()

# 処理結果の可視化
print("Visualizing processing results for NG Label 1:")
visualize_processing_results(processed_ng_images_label1, num_samples=1)

# 処理した画像数を表示
print(f"\nProcessed images count:")
print(f"NG Label 1: {len(processed_ng_images_label1)} images")
#print(f"NG Label 2: {len(processed_ng_images_label2)} images")
#print(f"NG Label 3: {len(processed_ng_images_label3)} images")
#print(f"OK: {len(processed_ok_images)} images")
```

この可視化コードの特徴：
1. 元画像と処理結果の比較が可能
2. 加工領域の強調表示により処理結果が分かりやすい
3. サンプル数の制限によりメモリ使用を抑制
4. 画像ごとに適切なタイトルと説明を付加

続けて#5以降のコードを生成しますか？

はい、#5から始めて、関数を整理し、モジュール性を高めながらコードを生成していきます。

```python
# 5. Canny+DoGによる欠陥検出

def apply_mask_to_image(image, mask):
    """
    画像にマスクを適用します

    引数：
        image (numpy.ndarray): 入力画像
        mask (numpy.ndarray): 適用するマスク

    戻り値：
        numpy.ndarray: マスク適用後の画像
    """
    return cv2.bitwise_and(image, image, mask=mask)

def extract_bright_dark_regions(image):
    """
    画像から明るい領域と暗い領域を抽出します

    引数：
        image (numpy.ndarray): 入力画像

    戻り値：
        tuple: (明るい領域のマスク, 暗い領域のマスク)
    """
    _, bright_mask = cv2.threshold(image, bright_threshold, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(image, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    return bright_mask, dark_mask

def calculate_gradient_magnitude(image):
    """
    画像の勾配強度を計算します

    引数：
        image (numpy.ndarray): 入力画像

    戻り値：
        numpy.ndarray: 正規化された勾配強度マップ
    """
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def detect_large_defects(image, mask):
    """
    Cannyエッジ検出で大きな鋳巣を検出します

    引数：
        image (numpy.ndarray): 入力画像
        mask (numpy.ndarray): マスク画像

    戻り値：
        numpy.ndarray: 検出結果
    """
    masked_image = apply_mask_to_image(image, mask)
    blurred_image = cv2.GaussianBlur(masked_image, canny_kernel_size, canny_sigma)
    edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
    
    # テクスチャ検出
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    laplacian_edges = np.uint8(np.abs(laplacian) > texture_threshold) * 255
    
    # 結果の統合と近接領域の結合
    combined_edges = cv2.bitwise_or(edges, laplacian_edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (canny_merge_distance, canny_merge_distance))
    return cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)

```

続きは次のメッセージで送ります。コードが長くなるため分割して送信します。

はい、Canny+DoGによる欠陥検出の続きを記載します。

```python
def apply_dog_filter(image, sigma_pairs):
    """
    マルチスケールDoGフィルタを適用します

    引数：
        image (numpy.ndarray): 入力画像
        sigma_pairs (list): 使用するシグマ値のペアのリスト

    戻り値：
        numpy.ndarray: DoGフィルタ適用後の画像
    """
    dog_results = []
    for sigma1, sigma2 in sigma_pairs:
        gaussian_1 = cv2.GaussianBlur(image, (dog_ksize, dog_ksize), sigma1)
        gaussian_2 = cv2.GaussianBlur(image, (dog_ksize, dog_ksize), sigma2)
        dog_result = gaussian_1 - gaussian_2
        normalized = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dog_results.append(normalized)
    
    return np.maximum.reduce(dog_results)

def calculate_contrast_mask(image, kernel_size):
    """
    画像のコントラストマスクを計算します

    引数：
        image (numpy.ndarray): 入力画像
        kernel_size (int): 局所領域のサイズ

    戻り値：
        numpy.ndarray: コントラストマスク
    """
    local_mean = cv2.blur(image, (kernel_size, kernel_size))
    intensity_diff = cv2.absdiff(image, local_mean)
    local_std = np.std(image)
    contrast_ratio = intensity_diff / (local_std + 1e-6)
    return (contrast_ratio > min_contrast_ratio).astype(np.uint8) * 255

def detect_small_defects(image, mask):
    """
    DoGフィルタで小さな鋳巣を検出します

    引数：
        image (numpy.ndarray): 入力画像
        mask (numpy.ndarray): マスク画像

    戻り値：
        numpy.ndarray: 検出結果
    """
    # マスク適用と前処理
    masked_image = apply_mask_to_image(image, mask)
    bright_mask, dark_mask = extract_bright_dark_regions(masked_image)
    
    # DoGフィルタの適用
    sigma_pairs = [(1.5, 3.5), (2.0, 4.0), (1.0, 2.5)]
    dog_result = apply_dog_filter(masked_image, sigma_pairs)
    
    # 動的閾値処理
    binary_dog = cv2.adaptiveThreshold(masked_image, 255, dynamic_method, 
                                     cv2.THRESH_BINARY_INV, dynamic_ksize, dynamic_c)
    
    # 勾配とコントラストの計算
    gradient_magnitude = calculate_gradient_magnitude(masked_image)
    contrast_mask = calculate_contrast_mask(masked_image, dynamic_ksize)
    
    # 結果の統合
    result = cv2.bitwise_and(dog_result, binary_dog)
    result = cv2.bitwise_and(result, contrast_mask)
    result = cv2.bitwise_and(result, mask)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dog_merge_distance, dog_merge_distance))
    merged_result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    # 明るい部分の追加
    return cv2.bitwise_or(merged_result, cv2.bitwise_and(bright_mask, mask))
```

続きは次のメッセージで送ります。

はい、処理部分の続きと可視化用のコードを記載します。

```python
def detect_defects(image, mask):
    """
    画像から大小の鋳巣を検出します

    引数：
        image (numpy.ndarray): 入力画像
        mask (numpy.ndarray): マスク画像

    戻り値：
        tuple: (統合結果, 大きな欠陥の検出結果, 小さな欠陥の検出結果)
    """
    # 大きな欠陥と小さな欠陥を検出
    large_defects = detect_large_defects(image, mask)
    small_defects = detect_small_defects(image, mask)
    
    # 結果の統合
    combined_result = cv2.bitwise_or(large_defects, small_defects)
    
    return combined_result, large_defects, small_defects

def process_images_for_defect_detection(processed_images):
    """
    処理済み画像から欠陥を検出します

    引数：
        processed_images (list): 処理済み画像のリスト

    戻り値：
        list: 欠陥検出結果のリスト
    """
    detected_images = []
    for binarized_image, processed_image, original_filename in processed_images:
        combined_result, large_defects, small_defects = detect_defects(
            processed_image, binarized_image)
        detected_images.append((
            original_filename,
            binarized_image,
            processed_image,
            combined_result,
            large_defects,
            small_defects
        ))
    return detected_images

# 欠陥検出の実行
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
        filename, mask, original, combined, large_defects, small_defects = detected_images[i]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 元画像
        axes[0,0].imshow(original, cmap='gray')
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        # マスク
        axes[0,1].imshow(mask, cmap='gray')
        axes[0,1].set_title('Mask')
        axes[0,1].axis('off')
        
        # 統合結果
        axes[0,2].imshow(combined, cmap='gray')
        axes[0,2].set_title('Combined Detection')
        axes[0,2].axis('off')
        
        # 大きな欠陥（Canny）
        axes[1,0].imshow(large_defects, cmap='gray')
        axes[1,0].set_title('Large Defects (Canny)')
        axes[1,0].axis('off')
        
        # 小さな欠陥（DoG）
        axes[1,1].imshow(small_defects, cmap='gray')
        axes[1,1].set_title('Small Defects (DoG)')
        axes[1,1].axis('off')
        
        # 重ね合わせ表示
        overlay = cv2.addWeighted(original, 0.7,
                                cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR), 0.3, 0)
        axes[1,2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1,2].set_title('Overlay')
        axes[1,2].axis('off')
        
        plt.suptitle(f'Defect Detection Results - {filename}', y=1.02)
        plt.tight_layout()
        plt.show()

# 検出結果の可視化
print("Visualizing defect detection results for NG Label 1:")
visualize_defect_detection(detected_ng_images_label1, num_samples=1)

# 処理した画像数を表示
print(f"\nProcessed images count:")
print(f"NG Label 1: {len(detected_ng_images_label1)} images")
#print(f"NG Label 2: {len(detected_ng_images_label2)} images")
#print(f"NG Label 3: {len(detected_ng_images_label3)} images")
#print(f"OK: {len(detected_ok_images)} images")
```

このコードでは：
1. 各機能を独立した関数に分割
2. 詳細なドキュメント文字列を追加
3. 処理と可視化を分離
4. 共通処理を再利用可能な関数として実装

続けて#6以降のコードを生成しますか？

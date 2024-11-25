```python
# 6. Canny+DoGによる欠陥検出

def detect_edges_and_texture(cropped_keyence_image, binarized_image):
    """
    Cannyエッジ検出とテクスチャ検出（大きな鋳巣用）
    """
    masked_image = cv2.bitwise_and(cropped_keyence_image, cropped_keyence_image, mask=binarized_image)
    blurred_image = cv2.GaussianBlur(masked_image, canny_kernel_size, canny_sigma)
    edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
    
    # テクスチャ検出
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    laplacian_edges = np.uint8(abs_laplacian > texture_threshold) * 255
    
    # エッジとテクスチャの統合
    combined_edges = cv2.bitwise_or(edges, laplacian_edges)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (canny_merge_distance, canny_merge_distance))
    merged_result = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
    
    return merged_result

def detect_defects_dog_dynamic(cropped_keyence_image, binarized_image):
    """
    DoGフィルタと動的閾値処理による検出（小さな鋳巣用）
    """
    # 元画像から直接、明るい部分と暗い部分を検出
    _, bright_mask = cv2.threshold(cropped_keyence_image, bright_threshold, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(cropped_keyence_image, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # マルチスケールDoGの適用
    dog_results = []
    sigma_pairs = [
        (1.5, 3.5),
        (2.0, 4.0),
        (1.0, 2.5)
    ]
    
    for sigma1, sigma2 in sigma_pairs:
        dog_result = difference_of_gaussian(cropped_keyence_image, dog_ksize, sigma1, sigma2)
        dog_result = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dog_results.append(dog_result)
    
    combined_dog = np.maximum.reduce(dog_results)
    
    # 動的閾値処理
    binary_dog = dynamic_threshold(cropped_keyence_image, dynamic_ksize, dynamic_method, dynamic_c)
    
    # 輝度の変化率を計算
    local_mean = cv2.blur(cropped_keyence_image, (dynamic_ksize, dynamic_ksize))
    intensity_diff = cv2.absdiff(cropped_keyence_image, local_mean)
    
    # 明るい領域と暗い領域の周辺の変化を強調
    gradient_x = cv2.Sobel(cropped_keyence_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(cropped_keyence_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # コントラスト比の計算
    local_std = np.std(cropped_keyence_image)
    contrast_ratio = intensity_diff / (local_std + 1e-6)
    contrast_mask = (contrast_ratio > min_contrast_ratio).astype(np.uint8) * 255
    
    # 輝度マスクと勾配マスクを組み合わせる
    combined_bright = cv2.bitwise_and(bright_mask, gradient_magnitude)
    combined_dark = cv2.bitwise_and(dark_mask, gradient_magnitude)
    combined_mask = cv2.bitwise_or(combined_bright, combined_dark)
    
    # DoGの結果と組み合わせる
    combined_result = cv2.bitwise_and(combined_mask, combined_dog)
    combined_result = cv2.bitwise_and(combined_result, contrast_mask)
    combined_result = cv2.bitwise_and(combined_result, binary_dog)
    
    # マスク適用
    masked_result = cv2.bitwise_and(combined_result, combined_result, mask=binarized_image)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dog_merge_distance, dog_merge_distance))
    merged_result = cv2.morphologyEx(masked_result, cv2.MORPH_CLOSE, kernel)
    
    # 最終的なマスクの作成（明るい部分と暗い部分の両方を含む）
    final_mask = cv2.bitwise_or(merged_result, cv2.bitwise_and(bright_mask, binarized_image))
    
    return final_mask

def difference_of_gaussian(img, ksize, sigma1, sigma2):
    """
    DoGフィルタの適用
    """
    # ガウシアンフィルタ適用
    gaussian_1 = cv2.GaussianBlur(img, (ksize, ksize), sigma1)
    gaussian_2 = cv2.GaussianBlur(img, (ksize, ksize), sigma2)
    
    # 2種のガウシアンフィルタ適用画像の差分
    dog = gaussian_1 - gaussian_2
    
    return dog

def dynamic_threshold(img, ksize, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, c=2):
    """
    適応的閾値処理
    """
    binary = cv2.adaptiveThreshold(img, 255, method, cv2.THRESH_BINARY_INV, ksize, c)
    return binary

def combine_defect_detection(image, mask):
    """
    CannyとDoGの結果を組み合わせて欠陥検出を行う
    """
    # 大きな鋳巣の検出（Canny）
    large_defects = detect_edges_and_texture(image, mask)
    
    # 小さな鋳巣の検出（DoG）
    small_defects = detect_defects_dog_dynamic(image, mask)
    
    # 結果の統合
    combined_result = cv2.bitwise_or(large_defects, small_defects)
    
    return combined_result, large_defects, small_defects  # 検出方法の区別のために個別の結果も返す

def process_images_for_defect_detection(binarized_images):
    """
    全画像に対して欠陥検出を実行。
    元のCanny/DoG形式に合わせて(binarized_image, defect_image, original_filename)の形式で出力
    """
    processed_images = []
    for binarized_image, cropped_keyence_image, original_filename in binarized_images:
        # Canny+DoGによる欠陥検出
        combined_result, _, _ = combine_defect_detection(cropped_keyence_image, binarized_image)
        # 元の形式に合わせて出力
        processed_images.append((binarized_image, combined_result, original_filename))
    return processed_images

# NGとOK画像に対して欠陥検出を実行
processed_ng_images_label1 = process_images_for_defect_detection(binarized_ng_images_label1)
processed_ng_images_label2 = process_images_for_defect_detection(binarized_ng_images_label2)
processed_ng_images_label3 = process_images_for_defect_detection(binarized_ng_images_label3)
processed_ok_images = process_images_for_defect_detection(binarized_ok_images)
```

主な変更点：
1. 元々のCanny処理とDoG処理を両方維持
2. 両方の処理に近接領域の統合を追加
3. 動的閾値処理をDoG処理に追加
4. 検出方法を区別するために、大きな鋳巣（Canny）と小さな鋳巣（DoG）の検出結果を個別に保持

この修正により、元々の処理を維持しながら、CannyとDoGの特徴を活かした統合的な検出が可能になります。

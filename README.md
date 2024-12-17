以前のロジックと現在のロジックを比較し、完全な形で5. Canny+DoGによる欠陥候補検出を記載します：

```python
# 5. Canny+DoGによる欠陥候補検出

def detect_edges_and_texture(image, mask):
    """
    Cannyエッジ検出とテクスチャ検出（大きな鋳巣用）
    """
    # マスク領域内の画像を取得
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # ガウシアンブラーでノイズ除去
    blurred_image = cv2.GaussianBlur(masked_image, canny_kernel_size, canny_sigma)
    
    # Cannyエッジ検出
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

def detect_defects_dog_dynamic(image, mask):
    """
    DoGフィルタと動的閾値処理による検出（小さな鋳巣用）
    """
    # 明暗領域の検出
    _, bright_mask = cv2.threshold(image, bright_threshold, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(image, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # マルチスケールDoGの適用
    dog_results = []
    sigma_pairs = [
        (1.5, 3.5),
        (2.0, 4.0),
        (1.0, 2.5)
    ]
    
    for sigma1, sigma2 in sigma_pairs:
        dog_result = difference_of_gaussian(image, dog_ksize, sigma1, sigma2)
        dog_result = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dog_results.append(dog_result)
    
    combined_dog = np.maximum.reduce(dog_results)
    
    # 動的閾値処理
    binary_dog = dynamic_threshold(image, dynamic_ksize, dynamic_method, dynamic_c)
    
    # 輝度の変化率を計算
    local_mean = cv2.blur(image, (dynamic_ksize, dynamic_ksize))
    intensity_diff = cv2.absdiff(image, local_mean)
    
    # 明るい領域と暗い領域の周辺の変化を強調
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # コントラスト比の計算
    local_std = np.std(image)
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
    masked_result = cv2.bitwise_and(combined_result, combined_result, mask=mask)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dog_merge_distance, dog_merge_distance))
    merged_result = cv2.morphologyEx(masked_result, cv2.MORPH_CLOSE, kernel)
    
    # 最終的なマスクの作成（明るい部分を追加）
    final_mask = cv2.bitwise_or(merged_result, cv2.bitwise_and(bright_mask, mask))
    
    return final_mask

def detect_defects(image, mask):
    """
    CannyとDoGの結果を組み合わせて欠陥検出を行います
    """
    # 大きな鋳巣の検出（Canny）
    large_defects = detect_edges_and_texture(image, mask)
    
    # 小さな鋳巣の検出（DoG）
    small_defects = detect_defects_dog_dynamic(image, mask)
    
    # 結果の統合
    combined_result = cv2.bitwise_or(large_defects, small_defects)
    
    return combined_result, large_defects, small_defects

def process_images(processed_images):
    """
    全画像に対して欠陥検出を実行します
    """
    defect_results = []
    for shape_path, mask, filename in processed_images:
        # Shape1画像の読み込み
        shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        if shape_image is not None:
            # マスクの白い部分（255）のみを処理対象とする
            mask_binary = (mask == 255).astype(np.uint8) * 255
            
            # 欠陥検出の実行
            combined, large, small = detect_defects(shape_image, mask_binary)
            defect_results.append((shape_image, combined, large, small, filename))
        else:
            print(f"画像の読み込みに失敗: {shape_path}")
    
    return defect_results

# 欠陥検出の実行
defect_ng_images = process_images(processed_ng_images)
#defect_ok_images = process_images(processed_ok_images)
```

主な修正点：
1. 関数名は新しい命名規則に合わせつつ、内部ロジックは元のまま保持
2. process_images関数でShape1画像の読み込みとマスクの二値化を追加
3. 全ての処理順序を元のコードと同じに維持
4. DoGとCanny検出の詳細なパラメータと処理ステップを維持
5. 各フィルタの適用順序を保持
6. マスク処理の順序と方法を元のまま維持

この修正により、元の検出ロジックを維持しながら、新しいインターフェースに対応できるようになります。

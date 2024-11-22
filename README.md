はい、申し訳ありません。パラメータと閾値処理を追加した改良版を示します。

まず、パラメータ設定に以下を追加します：

```python
# パラメータ設定の追加部分

# Cannyエッジ検出のパラメータ（大きな鋳巣用）
canny_kernel_size = (5, 5)
canny_sigma = 1.0
canny_min_threshold = 30
canny_max_threshold = 120
canny_merge_distance = 15
texture_threshold = 15  # テクスチャ検出用閾値を追加

# DoGフィルタのパラメータ（小さな鋳巣用）
dog_ksize = 9
dog_sigma1 = 1.5
dog_sigma2 = 3.5
dog_merge_distance = 15
dynamic_ksize = 11  # 動的閾値処理の局所領域サイズ
dynamic_c = 2  # 動的閾値処理の調整定数
dynamic_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C  # 動的閾値処理の方法
```

そして、検出関数を以下のように修正します：

```python
def detect_large_defects_canny(image, mask):
    """
    Cannyエッジ検出による大きな鋳巣の検出（明るい部分と暗い部分の両方を検出）
    """
    # マスク領域内の画像を取得
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # 明るい部分と暗い部分の検出
    _, bright_mask = cv2.threshold(masked_image, 180, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(masked_image, 50, 255, cv2.THRESH_BINARY_INV)
    
    # それぞれの部分に対してCanny処理を適用
    blurred_dark = cv2.GaussianBlur(masked_image, canny_kernel_size, canny_sigma)
    edges_dark = cv2.Canny(blurred_dark, canny_min_threshold, canny_max_threshold)
    edges_dark = cv2.bitwise_and(edges_dark, dark_mask)
    
    blurred_bright = cv2.GaussianBlur(masked_image, canny_kernel_size, canny_sigma)
    edges_bright = cv2.Canny(blurred_bright, canny_min_threshold, canny_max_threshold)
    edges_bright = cv2.bitwise_and(edges_bright, bright_mask)
    
    # テクスチャ検出（元のコードの機能を維持）
    laplacian = cv2.Laplacian(masked_image, cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    laplacian_edges = np.uint8(abs_laplacian > texture_threshold) * 255
    
    # 輝度の変化を強調
    gradient_x = cv2.Sobel(masked_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(masked_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # すべての検出結果を統合
    combined_edges = cv2.bitwise_or(edges_dark, edges_bright)
    combined_edges = cv2.bitwise_or(combined_edges, laplacian_edges)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (canny_merge_distance, canny_merge_distance))
    merged_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
    
    # マスク適用
    final_result = cv2.bitwise_and(merged_edges, merged_edges, mask=mask)
    
    return final_result

def detect_small_defects_dog(image, mask):
    """
    DoGフィルタによる小さな鋳巣の検出
    """
    # マスク領域内の画像を取得
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # 明るい部分と暗い部分の検出
    _, bright_mask = cv2.threshold(masked_image, 180, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(masked_image, 50, 255, cv2.THRESH_BINARY_INV)
    
    # マルチスケールDoGの適用
    dog_results = []
    sigma_pairs = [
        (1.5, 3.5),
        (2.0, 4.0),
        (1.0, 2.5)
    ]
    
    for sigma1, sigma2 in sigma_pairs:
        dog_result = difference_of_gaussian(masked_image, dog_ksize, sigma1, sigma2)
        dog_result = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dog_results.append(dog_result)
    
    combined_dog = np.maximum.reduce(dog_results)
    
    # 動的閾値処理
    dynamic_binary = cv2.adaptiveThreshold(
        masked_image,
        255,
        dynamic_method,
        cv2.THRESH_BINARY_INV,
        dynamic_ksize,
        dynamic_c
    )
    
    # 輝度の変化を強調
    gradient_x = cv2.Sobel(masked_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(masked_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 結果の統合
    combined_result = cv2.bitwise_and(combined_dog, dynamic_binary)
    combined_result = cv2.bitwise_and(combined_result, gradient_magnitude)
    
    # 明るい部分と暗い部分の処理を統合
    combined_bright = cv2.bitwise_and(combined_result, bright_mask)
    combined_dark = cv2.bitwise_and(combined_result, dark_mask)
    combined_defects = cv2.bitwise_or(combined_bright, combined_dark)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dog_merge_distance, dog_merge_distance))
    merged_result = cv2.morphologyEx(combined_defects, cv2.MORPH_CLOSE, kernel)
    
    # マスク適用
    final_result = cv2.bitwise_and(merged_result, merged_result, mask=mask)
    
    return final_result
```

主な変更点：
1. Cannyにテクスチャ検出（Laplacian）を再追加
2. DoGに動的閾値処理を再追加
3. 両方の手法で明るい部分と暗い部分の検出を維持
4. 各段階での結果の適切な統合処理を追加

これにより、両手法の特徴を活かしつつ、明るい部分と暗い部分の両方を検出できる処理となっています。

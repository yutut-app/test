はい、申し訳ありません。min_intensity_diffパラメータを追加し、同じロジックになるように修正します。

まず、パラメータの追加：
```python
# 2. パラメータの設定
# （既存のパラメータに追加）

# 輝度差の検出パラメータ
min_intensity_diff = 25  # 最小輝度差
min_contrast_ratio = 0.12  # 最小コントラスト比
```

そして、detect_defects_dog_dynamic関数を以下のように修正します：

```python
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
    
    # 輝度の変化率を計算
    local_mean = cv2.blur(cropped_keyence_image, (dynamic_ksize, dynamic_ksize))
    intensity_diff = cv2.absdiff(cropped_keyence_image, local_mean)
    
    # 輝度差に基づくフィルタリング
    intensity_mask = cv2.threshold(intensity_diff, min_intensity_diff, 255, cv2.THRESH_BINARY)[1]
    
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
    combined_result = cv2.bitwise_and(combined_result, intensity_mask)  # 輝度差マスクの追加
    
    # マスク適用
    masked_result = cv2.bitwise_and(combined_result, combined_result, mask=binarized_image)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dog_merge_distance, dog_merge_distance))
    merged_result = cv2.morphologyEx(masked_result, cv2.MORPH_CLOSE, kernel)
    
    # 最終的なマスクの作成（明るい部分と暗い部分の両方を含む）
    final_mask = cv2.bitwise_or(merged_result, cv2.bitwise_and(bright_mask, binarized_image))
    
    return final_mask
```

主な変更点：
1. min_intensity_diffパラメータを使用した輝度差マスクの生成を追加
2. 輝度差マスクを他のマスクと組み合わせる処理を追加

これで元のDoGロジックと同じ処理になります。

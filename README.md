申し訳ありません。承知しました。元画像（shape_images[base_name]）の白い部分を検出するために、二値化処理を行う前の状態で輝度の特徴を捉えるように修正します。

```python
# DoGフィルタのパラメータ
dog_ksize = 9
dog_sigma1 = 1.5
dog_sigma2 = 3.5

# 動的閾値処理のパラメータ
dynamic_ksize = 25
dynamic_c = 6
dynamic_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

# 輝度ベースの検出パラメータ
bright_threshold = 180  # 明るい部分の閾値
dark_threshold = 50    # 暗い部分の閾値
min_intensity_diff = 25
min_contrast_ratio = 0.12

# 近接領域の統合パラメータ
merge_distance = 15

def detect_defects_dog_dynamic(cropped_keyence_image, binarized_image):
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
    
    # マスク適用
    masked_result = cv2.bitwise_and(combined_result, combined_result, mask=binarized_image)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_distance, merge_distance))
    merged_result = cv2.morphologyEx(masked_result, cv2.MORPH_CLOSE, kernel)
    
    # 最終的なマスクの作成（明るい部分と暗い部分の両方を含む）
    final_mask = cv2.bitwise_or(merged_result, cv2.bitwise_and(bright_mask, binarized_image))
    
    return final_mask
```

主な変更点：

1. 元画像での輝度ベースの検出：
   - bright_thresholdで明るい部分を直接検出
   - dark_thresholdで暗い部分を直接検出

2. 輝度の変化率の検出を強化：
   - Sobelフィルタを使用して輝度の勾配を計算
   - 勾配の大きさを使用して輝度変化の大きい領域を強調

3. 明るい領域と暗い領域の統合：
   - 輝度マスクと勾配マスクを組み合わせて、特徴的な領域を抽出
   - 最終的に明るい部分と暗い部分の両方を含むマスクを生成

パラメータ調整のポイント：

1. 明るい部分の検出感度：
   - bright_thresholdを下げると、より多くの明るい領域を検出（150-200の範囲で調整）
   - 特に元画像の白い部分の輝度値を考慮して設定

2. 暗い部分の検出感度：
   - dark_thresholdを上げると、より多くの暗い領域を検出（30-70の範囲で調整）

3. 全体的な感度調整：
   - min_contrast_ratioを下げると、より多くの変化を検出
   - merge_distanceを調整して、検出された領域の統合範囲を制御

この改良により、元画像における白い部分と黒い部分の両方を検出し、それらを組み合わせて鋳巣の全体を捉えることができるはずです。画像の特性に応じて、特にbright_thresholdとdark_thresholdを適切に調整することが重要です。

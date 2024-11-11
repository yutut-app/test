なるほど、鋳巣の特徴として黒い部分と白い部分の両方を検出する必要があるということですね。これに対応するため、輝度の高い領域（白い部分）と低い領域（黒い部分）の両方を検出するようにコードを修正します。

```python
# DoGフィルタのパラメータは維持
dog_ksize = 9
dog_sigma1 = 1.5
dog_sigma2 = 3.5

# 動的閾値処理のパラメータ調整
dynamic_ksize = 25
dynamic_c = 6
dynamic_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

# 輝度差の検出パラメータ
min_intensity_diff_dark = 25  # 暗い部分の検出用
min_intensity_diff_bright = 20  # 明るい部分の検出用
min_contrast_ratio = 0.12

# 近接領域の統合パラメータ
merge_distance = 15

def detect_defects_dog_dynamic(cropped_keyence_image, binarized_image):
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
    
    # 動的閾値処理（暗い部分と明るい部分の両方）
    dynamic_result_dark = cv2.adaptiveThreshold(
        cropped_keyence_image, 255, dynamic_method, 
        cv2.THRESH_BINARY, dynamic_ksize, dynamic_c
    )
    
    dynamic_result_bright = cv2.adaptiveThreshold(
        cropped_keyence_image, 255, dynamic_method, 
        cv2.THRESH_BINARY_INV, dynamic_ksize, dynamic_c
    )
    
    # 輝度差の検出（暗い部分と明るい部分）
    local_mean = cv2.blur(cropped_keyence_image, (dynamic_ksize, dynamic_ksize))
    
    # 暗い部分の検出
    intensity_diff_dark = local_mean - cropped_keyence_image
    intensity_mask_dark = cv2.threshold(
        intensity_diff_dark, min_intensity_diff_dark, 255, cv2.THRESH_BINARY
    )[1]
    
    # 明るい部分の検出
    intensity_diff_bright = cropped_keyence_image - local_mean
    intensity_mask_bright = cv2.threshold(
        intensity_diff_bright, min_intensity_diff_bright, 255, cv2.THRESH_BINARY
    )[1]
    
    # コントラスト比の計算
    local_std = np.std(cropped_keyence_image)
    contrast_ratio_dark = intensity_diff_dark / (local_std + 1e-6)
    contrast_ratio_bright = intensity_diff_bright / (local_std + 1e-6)
    
    contrast_mask_dark = (contrast_ratio_dark > min_contrast_ratio).astype(np.uint8) * 255
    contrast_mask_bright = (contrast_ratio_bright > min_contrast_ratio).astype(np.uint8) * 255
    
    # 暗い部分と明るい部分の結果を統合
    dark_result = cv2.bitwise_and(combined_dog, dynamic_result_dark)
    dark_result = cv2.bitwise_and(dark_result, intensity_mask_dark)
    dark_result = cv2.bitwise_and(dark_result, contrast_mask_dark)
    
    bright_result = cv2.bitwise_and(combined_dog, dynamic_result_bright)
    bright_result = cv2.bitwise_and(bright_result, intensity_mask_bright)
    bright_result = cv2.bitwise_and(bright_result, contrast_mask_bright)
    
    # 暗い部分と明るい部分を統合
    combined_result = cv2.bitwise_or(dark_result, bright_result)
    
    # マスク適用
    masked_result = cv2.bitwise_and(combined_result, combined_result, mask=binarized_image)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_distance, merge_distance))
    merged_result = cv2.morphologyEx(masked_result, cv2.MORPH_CLOSE, kernel)
    
    return merged_result
```

主な変更点：

1. 輝度の異なる部分への対応：
   - 暗い部分（黒い領域）と明るい部分（白い領域）それぞれに対して検出処理を行う
   - それぞれの領域に適した閾値を設定

2. 動的閾値処理の拡張：
   - THRESH_BINARYとTHRESH_BINARY_INVの両方を使用
   - 暗い部分と明るい部分の両方に対応

3. 輝度差の検出方法の改良：
   - local_meanとの差分を両方向（より暗い、より明るい）で計算
   - それぞれに適した閾値を設定

パラメータ調整のポイント：

1. 白い部分の検出感度を調整する場合：
   - min_intensity_diff_brightを調整（小さくすると感度上昇）
   - dynamic_cを調整（小さくすると感度上昇）

2. 黒い部分の検出感度を調整する場合：
   - min_intensity_diff_darkを調整（小さくすると感度上昇）
   - dynamic_cを調整（小さくすると感度上昇）

3. 全体的な感度を調整する場合：
   - merge_distanceを調整（大きくすると統合範囲が広がる）
   - min_contrast_ratioを調整（小さくすると感度上昇）

この改良により、鋳巣の黒い部分と白い部分の両方を検出し、それらを一つの欠陥として統合することができるはずです。必要に応じて、各パラメータを調整して最適な検出結果を得ることができます。

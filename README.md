申し訳ありません。ご指摘ありがとうございます。元画像における白い部分（高輝度領域）を検出するように修正します。

```python
# DoGフィルタのパラメータ調整
dog_ksize = 9
dog_sigma1 = 1.5
dog_sigma2 = 3.5

# 動的閾値処理のパラメータ調整
dynamic_ksize = 25
dynamic_c = 6
dynamic_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

# 輝度差の検出パラメータ
min_intensity_diff_dark = 25  # 暗い部分（黒い領域）の検出用
min_intensity_diff_bright = 20  # 明るい部分（白い領域）の検出用
min_contrast_ratio = 0.12

# 元画像の輝度に関するパラメータ
bright_threshold = 180  # 高輝度領域の閾値
dark_threshold = 70    # 低輝度領域の閾値

# 近接領域の統合パラメータ
merge_distance = 15

def detect_defects_dog_dynamic(cropped_keyence_image, binarized_image):
    # 元画像から直接、高輝度領域と低輝度領域を検出
    _, bright_regions = cv2.threshold(cropped_keyence_image, bright_threshold, 255, cv2.THRESH_BINARY)
    _, dark_regions = cv2.threshold(cropped_keyence_image, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    
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
    dynamic_result = cv2.adaptiveThreshold(
        cropped_keyence_image, 255, dynamic_method, 
        cv2.THRESH_BINARY, dynamic_ksize, dynamic_c
    )
    
    # 輝度差の検出
    local_mean = cv2.blur(cropped_keyence_image, (dynamic_ksize, dynamic_ksize))
    intensity_diff = cv2.absdiff(cropped_keyence_image, local_mean)
    
    # コントラスト比の計算
    local_std = np.std(cropped_keyence_image)
    contrast_ratio = intensity_diff / (local_std + 1e-6)
    contrast_mask = (contrast_ratio > min_contrast_ratio).astype(np.uint8) * 255
    
    # DoGと動的閾値の結果を統合
    primary_result = cv2.bitwise_and(combined_dog, dynamic_result)
    
    # 高輝度領域と低輝度領域を考慮した最終結果の作成
    defect_regions = cv2.bitwise_or(bright_regions, dark_regions)
    defect_regions = cv2.bitwise_and(defect_regions, contrast_mask)
    
    # 全ての結果を統合
    combined_result = cv2.bitwise_or(primary_result, defect_regions)
    
    # マスク適用
    masked_result = cv2.bitwise_and(combined_result, combined_result, mask=binarized_image)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_distance, merge_distance))
    merged_result = cv2.morphologyEx(masked_result, cv2.MORPH_CLOSE, kernel)
    
    return merged_result
```

主な変更点：

1. 元画像の輝度に基づく検出：
   - 直接的な閾値処理で高輝度領域（bright_threshold以上）を検出
   - 同様に低輝度領域（dark_threshold以下）も検出
   - これにより、元画像の状態での白い部分と黒い部分を検出

2. パラメータの調整ポイント：

```python
# 高輝度/低輝度領域の検出感度調整用
bright_threshold = 180  # 小さくすると白い部分の検出感度が上がる（範囲：0-255）
dark_threshold = 70    # 大きくすると黒い部分の検出感度が上がる（範囲：0-255）
```

感度調整のガイドライン：

1. 白い部分（高輝度領域）の検出を調整する場合：
   - bright_thresholdを下げる：より多くの白い部分を検出（例：180→160）
   - bright_thresholdを上げる：より確実な白い部分のみを検出（例：180→200）

2. 黒い部分（低輝度領域）の検出を調整する場合：
   - dark_thresholdを上げる：より多くの黒い部分を検出（例：70→90）
   - dark_thresholdを下げる：より確実な黒い部分のみを検出（例：70→50）

3. 統合範囲を調整する場合：
   - merge_distanceを大きくする：より広い範囲を一つの欠陥として統合（例：15→20）
   - merge_distanceを小さくする：統合範囲を制限（例：15→10）

この改良により：
- 元画像の輝度値に基づいて直接的に白い部分と黒い部分を検出
- DoGフィルタと動的閾値処理の結果と組み合わせることで、より確実な欠陥検出が可能
- 近接領域の統合により、検出された領域を適切にグループ化

実際の画像で試しながら、bright_thresholdとdark_thresholdの値を調整することで、最適な検出結果を得ることができます。

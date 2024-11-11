# 追加のフィルタリングパラメータ
min_intensity_diff = 30  # 最小輝度差
min_contrast_ratio = 0.15  # 最小コントラスト比

def detect_defects_dog_dynamic(cropped_keyence_image, binarized_image):
    # DoGフィルタ適用
    dog_result = difference_of_gaussian(cropped_keyence_image, dog_ksize, dog_sigma1, dog_sigma2)
    
    # DoG結果を8ビット unsigned int に変換
    dog_result = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 動的閾値処理
    dynamic_result = dynamic_threshold(cropped_keyence_image, dynamic_ksize, dynamic_method, dynamic_c)
    
    # 輝度差に基づくフィルタリング
    local_mean = cv2.blur(cropped_keyence_image, (dynamic_ksize, dynamic_ksize))
    intensity_diff = cv2.absdiff(cropped_keyence_image, local_mean)
    intensity_mask = cv2.threshold(intensity_diff, min_intensity_diff, 255, cv2.THRESH_BINARY)[1]
    
    # コントラスト比に基づくフィルタリング
    local_std = np.std(cropped_keyence_image)
    contrast_ratio = intensity_diff / (local_std + 1e-6)
    contrast_mask = (contrast_ratio > min_contrast_ratio).astype(np.uint8) * 255
    
    # 全ての条件を組み合わせる
    combined_result = cv2.bitwise_and(dog_result, dynamic_result)
    combined_result = cv2.bitwise_and(combined_result, intensity_mask)
    combined_result = cv2.bitwise_and(combined_result, contrast_mask)
    
    # マスク適用
    masked_result = cv2.bitwise_and(combined_result, combined_result, mask=binarized_image)
    
    return masked_result

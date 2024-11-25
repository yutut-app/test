def detect_edges_and_texture(cropped_keyence_image, binarized_image):
    """
    Cannyエッジ検出とテクスチャ検出（大きな鋳巣用）
    """
    # マスク領域内の画像を取得
    masked_image = cv2.bitwise_and(cropped_keyence_image, cropped_keyence_image, mask=binarized_image)
    
    # 元画像から直接、明るい部分と暗い部分を検出
    _, bright_mask = cv2.threshold(cropped_keyence_image, bright_threshold, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(cropped_keyence_image, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Cannyエッジ検出
    blurred_image = cv2.GaussianBlur(masked_image, canny_kernel_size, canny_sigma)
    edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
    
    # テクスチャ検出
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    laplacian_edges = np.uint8(abs_laplacian > texture_threshold) * 255
    
    # 勾配強度の計算（輝度変化の大きい領域の検出用）
    gradient_x = cv2.Sobel(masked_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(masked_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 輝度マスクと勾配マスクを組み合わせる
    combined_bright = cv2.bitwise_and(bright_mask, gradient_magnitude)
    combined_dark = cv2.bitwise_and(dark_mask, gradient_magnitude)
    
    # エッジ、テクスチャ、輝度マスクの統合
    combined_edges = cv2.bitwise_or(edges, laplacian_edges)
    combined_edges = cv2.bitwise_or(combined_edges, combined_bright)
    combined_edges = cv2.bitwise_or(combined_edges, combined_dark)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (canny_merge_distance, canny_merge_distance))
    merged_result = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
    
    # マスク適用
    masked_result = cv2.bitwise_and(merged_result, merged_result, mask=binarized_image)
    
    # 最終的なマスクの作成（明るい部分と暗い部分の両方を含む）
    final_mask = cv2.bitwise_or(masked_result, cv2.bitwise_and(bright_mask, binarized_image))
    
    return final_mask

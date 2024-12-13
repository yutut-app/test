# 5. Canny+DoGによる欠陥検出

def apply_canny_detection(image, mask):
    """
    Cannyエッジ検出を用いて大きな鋳巣を検出します
    
    引数:
        image (numpy.ndarray): 入力画像
        mask (numpy.ndarray): マスク画像
        
    戻り値:
        numpy.ndarray: エッジ検出結果
    """
    # マスク領域内の画像を取得
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # ガウシアンブラーでノイズ除去
    blurred_image = cv2.GaussianBlur(masked_image, canny_kernel_size, canny_sigma)
    
    # Cannyエッジ検出
    edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
    
    # テクスチャ検出
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    laplacian_edges = np.uint8(np.absolute(laplacian) > texture_threshold) * 255
    
    # エッジとテクスチャの統合
    combined_edges = cv2.bitwise_or(edges, laplacian_edges)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                     (canny_merge_distance, canny_merge_distance))
    merged_result = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
    
    return merged_result

def apply_dog_filter(image, ksize, sigma1, sigma2):
    """
    DoGフィルタを適用します
    
    引数:
        image (numpy.ndarray): 入力画像
        ksize (int): カーネルサイズ
        sigma1 (float): 1つ目のガウシアンのシグマ
        sigma2 (float): 2つ目のガウシアンのシグマ
        
    戻り値:
        numpy.ndarray: DoGフィルタ適用結果
    """
    gaussian_1 = cv2.GaussianBlur(image, (ksize, ksize), sigma1)
    gaussian_2 = cv2.GaussianBlur(image, (ksize, ksize), sigma2)
    return gaussian_1 - gaussian_2

def apply_dynamic_threshold(image):
    """
    動的閾値処理を適用します
    
    引数:
        image (numpy.ndarray): 入力画像
        
    戻り値:
        numpy.ndarray: 二値化結果
    """
    return cv2.adaptiveThreshold(image, 255, dynamic_method, cv2.THRESH_BINARY_INV, 
                               dynamic_ksize, dynamic_c)

def calculate_contrast_mask(image):
    """
    コントラストに基づくマスクを生成します
    
    引数:
        image (numpy.ndarray): 入力画像
        
    戻り値:
        numpy.ndarray: コントラストマスク
    """
    # 局所的な平均との差を計算
    local_mean = cv2.blur(image, (dynamic_ksize, dynamic_ksize))
    intensity_diff = cv2.absdiff(image, local_mean)
    
    # コントラスト比を計算
    local_std = np.std(image)
    contrast_ratio = intensity_diff / (local_std + 1e-6)
    
    return (contrast_ratio > min_contrast_ratio).astype(np.uint8) * 255

def apply_dog_detection(image, mask):
    """
    DoGフィルタを用いて小さな鋳巣を検出します
    
    引数:
        image (numpy.ndarray): 入力画像
        mask (numpy.ndarray): マスク画像
        
    戻り値:
        numpy.ndarray: 検出結果
    """
    # 明暗領域の検出
    _, bright_mask = cv2.threshold(image, bright_threshold, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(image, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # マルチスケールDoGの適用
    dog_results = []
    sigma_pairs = [(1.5, 3.5), (2.0, 4.0), (1.0, 2.5)]
    
    for sigma1, sigma2 in sigma_pairs:
        dog_result = apply_dog_filter(image, dog_ksize, sigma1, sigma2)
        normalized = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dog_results.append(normalized)
    
    combined_dog = np.maximum.reduce(dog_results)
    
    # 各種マスクの生成
    binary_dog = apply_dynamic_threshold(image)
    contrast_mask = calculate_contrast_mask(image)
    gradient_magnitude = calculate_gradient_magnitude(image)
    
    # マスクの統合
    bright_region = cv2.bitwise_and(bright_mask, gradient_magnitude)
    dark_region = cv2.bitwise_and(dark_mask, gradient_magnitude)
    combined_mask = cv2.bitwise_or(bright_region, dark_region)
    
    # 結果の統合
    result = cv2.bitwise_and(combined_mask, combined_dog)
    result = cv2.bitwise_and(result, contrast_mask)
    result = cv2.bitwise_and(result, binary_dog)
    result = cv2.bitwise_and(result, mask)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                     (dog_merge_distance, dog_merge_distance))
    merged_result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    # 明るい領域の追加
    final_result = cv2.bitwise_or(merged_result, cv2.bitwise_and(bright_mask, mask))
    
    return final_result

def calculate_gradient_magnitude(image):
    """
    画像の勾配強度を計算します
    
    引数:
        image (numpy.ndarray): 入力画像
        
    戻り値:
        numpy.ndarray: 勾配強度画像
    """
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def detect_defects(image, mask):
    """
    CannyとDoGを組み合わせて欠陥検出を行います
    
    引数:
        image (numpy.ndarray): 入力画像
        mask (numpy.ndarray): マスク画像
        
    戻り値:
        tuple: (統合結果, Canny結果, DoG結果)
    """
    large_defects = apply_canny_detection(image, mask)
    small_defects = apply_dog_detection(image, mask)
    combined_result = cv2.bitwise_or(large_defects, small_defects)
    
    return combined_result, large_defects, small_defects

def process_images(processed_images):
    """
    全画像に対して欠陥検出を実行します
    
    引数:
        processed_images (list): 処理済み画像のリスト
        
    戻り値:
        list: (画像, 検出結果, Canny結果, DoG結果, ファイル名)のリスト
    """
    defect_results = []
    for shape_image, mask, filename in processed_images:
        combined, large, small = detect_defects(shape_image, mask)
        defect_results.append((shape_image, combined, large, small, filename))
    return defect_results

# 欠陥検出の実行
defect_ng_images = process_images(processed_ng_images)
#defect_ok_images = process_images(processed_ok_images)




def visualize_defect_detection(defect_results, num_samples=1):
    """
    欠陥検出結果を可視化します
    
    引数:
        defect_results (list): 欠陥検出結果のリスト
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(defect_results))
    
    for i in range(num_samples):
        shape_image, combined, large, small, filename = defect_results[i]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # 元画像
        axes[0, 0].imshow(shape_image, cmap='gray')
        axes[0, 0].set_title('Original Shape Image')
        axes[0, 0].axis('off')
        
        # Cannyエッジ検出結果
        axes[0, 1].imshow(large, cmap='gray')
        axes[0, 1].set_title('Large Defects (Canny)')
        axes[0, 1].axis('off')
        
        # DoG検出結果
        axes[1, 0].imshow(small, cmap='gray')
        axes[1, 0].set_title('Small Defects (DoG)')
        axes[1, 0].axis('off')
        
        # 統合結果
        axes[1, 1].imshow(combined, cmap='gray')
        axes[1, 1].set_title('Combined Result')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Defect Detection Results: {filename}')
        plt.tight_layout()
        plt.show()

# 検出結果の可視化
print("Visualizing defect detection results for NG images:")
visualize_defect_detection(defect_ng_images, num_samples=1)
#print("\nVisualizing defect detection results for OK images:")
#visualize_defect_detection(defect_ok_images, num_samples=1)

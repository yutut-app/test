#6のコードを修正し、DoGと同じ形式で画像を扱うように変更します：

```python
# 6. Canny+DoGによる欠陥検出

# [前のパラメータ設定部分は変更なし]

def detect_large_defects_canny(image, mask):
    """
    Cannyエッジ検出による大きな鋳巣の検出
    """
    # マスク領域内の画像を取得
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # ガウシアンフィルタでノイズ除去
    blurred = cv2.GaussianBlur(masked_image, canny_kernel_size, canny_sigma)
    
    # Cannyエッジ検出
    edges = cv2.Canny(blurred, canny_min_threshold, canny_max_threshold)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (canny_merge_distance, canny_merge_distance))
    merged_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return merged_edges

def detect_small_defects_dog(image, mask):
    """
    DoGフィルタによる小さな鋳巣の検出
    """
    # マスク領域内の画像を取得
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
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
    
    # DoG結果の統合
    combined_dog = np.maximum.reduce(dog_results)
    
    # 閾値処理
    _, binary_dog = cv2.threshold(combined_dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dog_merge_distance, dog_merge_distance))
    merged_dog = cv2.morphologyEx(binary_dog, cv2.MORPH_CLOSE, kernel)
    
    return merged_dog

def combine_defect_detection(image, mask):
    """
    CannyとDoGの結果を組み合わせて欠陥検出を行う
    """
    # 大きな鋳巣の検出（Canny）
    large_defects = detect_large_defects_canny(image, mask)
    
    # 小さな鋳巣の検出（DoG）
    small_defects = detect_small_defects_dog(image, mask)
    
    # サイズによるフィルタリング
    large_defects_filtered = filter_by_size(large_defects, min_large_defect_size, max_large_defect_size)
    small_defects_filtered = filter_by_size(small_defects, min_small_defect_size, max_small_defect_size)
    
    # 結果の統合
    combined_result = cv2.bitwise_or(large_defects_filtered, small_defects_filtered)
    
    # マスク適用
    final_result = cv2.bitwise_and(combined_result, combined_result, mask=mask)
    
    return final_result

def filter_by_size(binary_image, min_size, max_size):
    """
    サイズに基づいて欠陥候補をフィルタリング
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    
    filtered_image = np.zeros_like(binary_image)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_size <= area <= max_size:
            filtered_image[labels == i] = 255
    
    return filtered_image

def process_images_for_defect_detection(binarized_images):
    """
    全画像に対して欠陥検出を実行
    """
    processed_images = []
    for binarized_image, cropped_keyence_image, original_filename in binarized_images:
        defect_image = combine_defect_detection(cropped_keyence_image, binarized_image)
        processed_images.append((binarized_image, defect_image, original_filename))
    return processed_images

# NGとOK画像に対して欠陥検出を実行
processed_ng_images_label1 = process_images_for_defect_detection(binarized_ng_images_label1)
processed_ng_images_label2 = process_images_for_defect_detection(binarized_ng_images_label2)
processed_ng_images_label3 = process_images_for_defect_detection(binarized_ng_images_label3)
processed_ok_images = process_images_for_defect_detection(binarized_ok_images)
```

主な変更点：
1. 画像の読み込みと表示部分を削除し、DoGと同じ形式でデータを扱うように変更
2. process_images_for_defect_detectionの出力形式をDoGと同じに統一
3. 全ての画像処理をグレースケールで統一
4. 出力形式を(binarized_image, defect_image, original_filename)に統一

これにより、DoGの処理と同じ形式で画像を扱えるようになり、TypeError を回避できます。

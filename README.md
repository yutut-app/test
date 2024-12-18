```python
# 7. 欠陥候補のフィルタリング

def measure_region_properties(region, label, detection_method):
    """
    領域の特徴量を測定します
    
    引数:
        region: regionpropsで得られた領域情報
        label (int): 欠陥のラベル番号
        detection_method (str): 検出方法('canny'または'dog')
    
    戻り値:
        dict: 測定した特徴量情報
    """
    y, x = region.bbox[0], region.bbox[1]
    h, w = region.bbox[2] - y, region.bbox[3] - x
    
    return {
        'label': label,
        'x': x, 'y': y, 'width': w, 'height': h,
        'area': region.area,
        'centroid_y': region.centroid[0],
        'centroid_x': region.centroid[1],
        'perimeter': region.perimeter,
        'eccentricity': region.eccentricity,
        'orientation': region.orientation,
        'major_axis_length': region.major_axis_length,
        'minor_axis_length': region.minor_axis_length,
        'solidity': region.solidity,
        'extent': region.extent,
        'aspect_ratio': max(w, h) / min(w, h) if min(w, h) > 0 else 0,
        'max_length': max(w, h),
        'detection_method': detection_method
    }

def filter_defects_by_size(binary_image, mask_edges, min_size, max_size, detection_method, start_label):
    """
    サイズに基づいて欠陥候補をフィルタリングします
    
    引数:
        binary_image (ndarray): 二値化画像
        mask_edges (ndarray): マスクのエッジ
        min_size (int): 最小サイズ
        max_size (int): 最大サイズ
        detection_method (str): 検出方法
        start_label (int): ラベルの開始番号
    
    戻り値:
        list: フィルタリングされた欠陥情報のリスト
    """
    # マスクエッジ部分を除外
    filtered_image = binary_image.copy()
    filtered_image[mask_edges > 0] = 0
    
    # ラベリング処理
    labels = measure.label(filtered_image, connectivity=2)
    defects = []
    
    # 各領域の処理
    for region in measure.regionprops(labels):
        if min_size <= region.area <= max_size:
            label = start_label + len(defects) + 1
            defect_info = measure_region_properties(region, label, detection_method)
            defects.append(defect_info)
    
    return defects

def filter_and_measure_defects(combined_result, large_defects, small_defects, mask):
    """
    CannyとDoGの結果それぞれに対して適切なサイズ範囲でフィルタリングを行います
    
    引数:
        combined_result (ndarray): 統合された検出結果
        large_defects (ndarray): Cannyによる検出結果
        small_defects (ndarray): DoGによる検出結果
        mask (ndarray): マスク画像
    
    戻り値:
        tuple: フィルタリングされた欠陥情報のリストと二値化画像
    """
    # マスクエッジの作成
    mask_edges = create_mask_edge_margin(mask, mask_edge_margin)
    
    # 大きな欠陥（Canny）の処理
    binary_large = (large_defects > 0).astype(np.uint8)
    large_defects_info = filter_defects_by_size(
        binary_large, mask_edges,
        min_large_defect_size, max_large_defect_size,
        'canny', 0
    )
    
    # 小さな欠陥（DoG）の処理
    binary_small = (small_defects > 0).astype(np.uint8)
    small_defects_info = filter_defects_by_size(
        binary_small, mask_edges,
        min_small_defect_size, max_small_defect_size,
        'dog', len(large_defects_info)
    )
    
    return large_defects_info + small_defects_info, binary_large, binary_small

def process_images_for_filtering(completed_images):
    """
    画像群に対してフィルタリング処理を実行します
    
    引数:
        completed_images (list): エッジ処理済み画像のリスト
    
    戻り値:
        list: フィルタリング結果のリスト
    """
    filtered_images = []
    for mask, large_defects, small_defects, original_filename in completed_images:
        # フィルタリング処理
        defects, binary_large, binary_small = filter_and_measure_defects(
            None, large_defects, small_defects, mask
        )
        filtered_images.append((mask, defects, binary_large, binary_small, original_filename))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images = process_images_for_filtering(completed_ng_images)
filtered_ok_images = process_images_for_filtering(completed_ok_images)
```

可視化のコードは次のメッセージで送ります。

```python
# 7. 欠陥候補のフィルタリング

def measure_defect_properties(region, label, detection_method):
    """
    領域の特徴量を計測します
    
    引数:
        region: measure.regionpropsの領域オブジェクト
        label (int): 欠陥候補のラベル番号 
        detection_method (str): 検出手法('canny'または'dog')
        
    戻り値:
        dict: 欠陥候補の特徴量情報
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

def filter_defects_by_size(binary_image, min_size, max_size, detection_method, start_label):
    """
    サイズに基づいて欠陥候補をフィルタリングします
    
    引数:
        binary_image (ndarray): 二値化画像
        min_size (int): 最小サイズ
        max_size (int): 最大サイズ
        detection_method (str): 検出手法
        start_label (int): ラベル開始番号
        
    戻り値:
        list: フィルタリングされた欠陥情報のリスト
    """
    defects = []
    labels = measure.label(binary_image, connectivity=2)
    
    for region in measure.regionprops(labels):
        if min_size <= region.area <= max_size:
            defect_info = measure_defect_properties(
                region, 
                start_label + len(defects) + 1,
                detection_method
            )
            defects.append(defect_info)
            
    return defects

def filter_and_measure_defects(edge_image, large_defects, small_defects, mask):
    """
    CannyとDoGの結果それぞれに対して、サイズベースのフィルタリングと特徴量計測を行います
    
    引数:
        edge_image (ndarray): エッジ画像
        large_defects (ndarray): Cannyによる検出結果
        small_defects (ndarray): DoGによる検出結果
        mask (ndarray): マスク画像
        
    戻り値:
        tuple: フィルタリング後の欠陥リストと、大小の欠陥画像
    """
    # マスクエッジの作成
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    
    # 大きな欠陥（Canny）の処理
    binary_large = (large_defects > 0).astype(np.uint8)
    binary_large[mask_edges_with_margin > 0] = 0
    large_filtered = filter_defects_by_size(
        binary_large, 
        min_large_defect_size, 
        max_large_defect_size, 
        'canny', 
        0
    )
    
    # 小さな欠陥（DoG）の処理
    binary_small = (small_defects > 0).astype(np.uint8)
    binary_small[mask_edges_with_margin > 0] = 0
    small_filtered = filter_defects_by_size(
        binary_small,
        min_small_defect_size,
        max_small_defect_size,
        'dog',
        len(large_filtered)
    )
    
    return large_filtered + small_filtered, binary_large, binary_small

def process_images_for_filtering(completed_images):
    """
    画像群に対して欠陥候補のフィルタリングを実行します
    
    引数:
        completed_images (list): エッジ処理済み画像のリスト
        
    戻り値:
        list: フィルタリング結果のリスト
    """
    filtered_images = []
    for mask, large_defects, small_defects, original_filename in completed_images:
        defects, binary_large, binary_small = filter_and_measure_defects(
            None,  # edge_imageは使用しないのでNone
            large_defects,
            small_defects,
            mask
        )
        filtered_images.append((mask, binary_large, binary_small, defects, original_filename))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images = process_images_for_filtering(completed_ng_images)
filtered_ok_images = process_images_for_filtering(completed_ok_images)
```

続きのビジュアライゼーション部分を次のメッセージで送ります。

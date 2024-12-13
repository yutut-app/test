def extract_region_features(region):
    """
    領域から特徴量を抽出します
    
    引数:
        region (RegionProperties): skimageのregionpropsによる領域情報
        
    戻り値:
        dict: 抽出された特徴量の辞書
    """
    y, x = region.bbox[0], region.bbox[1]
    h, w = region.bbox[2] - y, region.bbox[3] - x
    max_length = max(w, h)
    
    # 検出方法の判定を修正
    if min_large_defect_size <= max_length <= max_large_defect_size:
        detection_method = 'canny'
    elif min_small_defect_size <= max_length <= max_small_defect_size:
        detection_method = 'dog'
    else:
        detection_method = None
    
    return {
        'label': region.label,
        'x': x,
        'y': y,
        'width': w,
        'height': h,
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
        'max_length': max_length,
        'detection_method': detection_method
    }

def create_binary_edge_image(edge_image, mask):
    """
    エッジ画像を二値化し、マスクエッジ部分を除外します
    
    引数:
        edge_image (numpy.ndarray): エッジ画像
        mask (numpy.ndarray): マスク画像
        
    戻り値:
        numpy.ndarray: 二値化されたエッジ画像
    """
    mask_edges = create_mask_edge_margin(mask)
    binary_edge_image = (edge_image > 0).astype(np.uint8)
    binary_edge_image[mask_edges > 0] = 0
    return binary_edge_image

def filter_defects(edge_image, mask):
    """
    欠陥候補をフィルタリングし、特徴量を計測します
    
    引数:
        edge_image (numpy.ndarray): エッジ画像
        mask (numpy.ndarray): マスク画像
        
    戻り値:
        list: 欠陥情報の辞書のリスト
    """
    # エッジ画像の二値化とマスクエッジの除外
    binary_edge_image = create_binary_edge_image(edge_image, mask)
    
    # 連結成分のラベリング
    labels = measure.label(binary_edge_image, connectivity=2)
    
    # 各領域の特徴量を計測
    defects = []
    for region in measure.regionprops(labels):
        defect_info = extract_region_features(region)
        # 検出方法がNoneでない（有効なサイズ範囲内の）欠陥のみを追加
        if defect_info['detection_method'] is not None:
            defects.append(defect_info)
    
    return defects

def process_completed_edges(completed_results):
    """
    補完済みエッジに対してフィルタリングと特徴量計測を実行します
    
    引数:
        completed_results (list): 補完済みエッジのリスト
        
    戻り値:
        list: フィルタリング結果のリスト
        [(画像, 統合結果, Canny結果の欠陥, DoG結果の欠陥, ファイル名)]
    """
    filtered_results = []
    
    for shape_image, completed_combined, completed_large, completed_small, filename in completed_results:
        # 各結果に対してフィルタリングを実行
        combined_defects = filter_defects(completed_combined, shape_image)
        large_defects = filter_defects(completed_large, shape_image)
        small_defects = filter_defects(completed_small, shape_image)
        
        filtered_results.append(
            (shape_image, completed_combined, large_defects, small_defects, filename)
        )
    
    return filtered_results

# フィルタリングの実行
filtered_ng_results = process_completed_edges(completed_ng_images)
#filtered_ok_results = process_completed_edges(completed_ok_images)

def visualize_filtered_defects(filtered_results, num_samples=1):
    """
    フィルタリング結果を可視化します
    
    引数:
        filtered_results (list): フィルタリング結果のリスト
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(filtered_results))
    
    for i in range(num_samples):
        shape_image, combined, large_defects, small_defects, filename = filtered_results[i]
        
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(shape_image, cmap='gray')
        
        # 大きな欠陥を赤色で表示（Canny検出）
        for defect in large_defects:
            if defect['detection_method'] == 'canny':
                rect = plt.Rectangle(
                    (defect['x'], defect['y']),
                    defect['width'],
                    defect['height'],
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                )
                ax.add_patch(rect)
                ax.text(
                    defect['x'],
                    defect['y'],
                    f"L{defect['label']}",
                    color='red',
                    fontsize=12
                )
        
        # 小さな欠陥を青色で表示（DoG検出）
        for defect in small_defects:
            if defect['detection_method'] == 'dog':
                rect = plt.Rectangle(
                    (defect['x'], defect['y']),
                    defect['width'],
                    defect['height'],
                    fill=False,
                    edgecolor='blue',
                    linewidth=2
                )
                ax.add_patch(rect)
                ax.text(
                    defect['x'],
                    defect['y'],
                    f"S{defect['label']}",
                    color='blue',
                    fontsize=12
                )
        
        # 凡例を追加
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='red', label='Large Defects (Canny)'),
            Patch(facecolor='none', edgecolor='blue', label='Small Defects (DoG)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f'Filtered Defects: {filename}')
        plt.axis('off')
        plt.show()

# フィルタリングの実行
filtered_ng_results = process_completed_edges(completed_ng_images)
#filtered_ok_results = process_completed_edges(completed_ok_images)

# フィルタリング結果の可視化
print("Visualizing filtered defects for NG images:")
visualize_filtered_defects(filtered_ng_results, num_samples=1)
#print("\nVisualizing filtered defects for OK images:")
#visualize_filtered_defects(filtered_ok_results, num_samples=1)


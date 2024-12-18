```python
# 7. 欠陥候補のフィルタリング

def measure_defect_properties(region, label, detection_method):
    """
    欠陥領域の特徴量を計測します
    
    引数:
        region: 欠陥領域のRegionPropsオブジェクト
        label (int): 欠陥のラベル番号
        detection_method (str): 検出方法（'canny' or 'dog'）
    
    戻り値:
        dict: 欠陥の特徴量情報
    """
    y, x = region.bbox[0], region.bbox[1]
    h, w = region.bbox[2] - y, region.bbox[3] - x
    
    return {
        'label': label,
        'x': x, 'y': y, 
        'width': w, 'height': h,
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

def filter_defects_by_size(binary_image, min_size, max_size):
    """
    サイズに基づいて欠陥をフィルタリングします
    
    引数:
        binary_image (ndarray): 二値化画像
        min_size (int): 最小サイズ
        max_size (int): 最大サイズ
    
    戻り値:
        list: フィルタリングされた領域のリスト
    """
    labels = measure.label(binary_image, connectivity=2)
    filtered_regions = []
    
    for region in measure.regionprops(labels):
        if min_size <= region.area <= max_size:
            filtered_regions.append(region)
            
    return filtered_regions

def filter_and_measure_defects(large_defects, small_defects, mask):
    """
    CannyとDoGの結果をサイズでフィルタリングし、特徴量を計測します
    
    引数:
        large_defects (ndarray): Canny検出結果
        small_defects (ndarray): DoG検出結果
        mask (ndarray): マスク画像
    
    戻り値:
        tuple: 欠陥情報リストと処理後の画像
    """
    # マスクエッジの除外領域を作成
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    
    # Canny検出結果の処理
    binary_large = (large_defects > 0).astype(np.uint8)
    binary_large[mask_edges_with_margin > 0] = 0
    large_regions = filter_defects_by_size(binary_large, min_large_defect_size, max_large_defect_size)
    
    # DoG検出結果の処理
    binary_small = (small_defects > 0).astype(np.uint8)
    binary_small[mask_edges_with_margin > 0] = 0
    small_regions = filter_defects_by_size(binary_small, min_small_defect_size, max_small_defect_size)
    
    # 特徴量の計測
    defects = []
    current_label = 1
    
    # Canny結果の特徴量計測
    for region in large_regions:
        defect_info = measure_defect_properties(region, current_label, 'canny')
        defects.append(defect_info)
        current_label += 1
    
    # DoG結果の特徴量計測
    for region in small_regions:
        defect_info = measure_defect_properties(region, current_label, 'dog')
        defects.append(defect_info)
        current_label += 1
    
    return defects, binary_large, binary_small

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
        defects, filtered_large, filtered_small = filter_and_measure_defects(
            large_defects, small_defects, mask
        )
        filtered_images.append((mask, filtered_large, filtered_small, defects, original_filename))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images = process_images_for_filtering(completed_ng_images)
filtered_ok_images = process_images_for_filtering(completed_ok_images)
```

```python
# フィルタリング結果の可視化

def create_defect_visualization(mask, defects_image, defects_info=None, show_labels=False):
    """
    欠陥検出結果の可視化画像を作成します
    
    引数:
        mask (ndarray): マスク画像
        defects_image (tuple): (大きな欠陥画像, 小さな欠陥画像)
        defects_info (list): 欠陥情報のリスト（ラベル表示用）
        show_labels (bool): ラベルを表示するかどうか
        
    戻り値:
        ndarray: 可視化用のRGB画像
    """
    large_defects, small_defects = defects_image
    visualization = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # マスク領域をグレーで表示
    visualization[mask > 0] = [200, 200, 200]
    
    # 欠陥を色分けして表示
    visualization[large_defects > 0] = [255, 0, 0]  # Canny結果を赤で表示
    visualization[small_defects > 0] = [0, 0, 255]  # DoG結果を青で表示
    
    # ラベルの表示
    if show_labels and defects_info:
        visualization = visualization.copy()
        for defect in defects_info:
            x, y = int(defect['x']), int(defect['y'])
            color = (255, 0, 0) if defect['detection_method'] == 'canny' else (0, 0, 255)
            cv2.putText(
                visualization, 
                str(defect['label']),
                (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                1
            )
            
    return visualization

def visualize_filtering_results(processed_images, filtered_images, pair_index):
    """
    フィルタリング前後の結果を可視化します
    
    引数:
        processed_images (list): フィルタリング前の画像リスト
        filtered_images (list): フィルタリング後の画像リスト
        pair_index (int): 表示するペアのインデックス
    """
    if not processed_images or not filtered_images or pair_index >= len(processed_images):
        print("指定されたインデックスの画像が存在しません")
        return
    
    # フィルタリング前のデータ取得
    mask_before, _, large_before, small_before, filename = processed_images[pair_index]
    # フィルタリング後のデータ取得
    mask_after, filtered_large, filtered_small, defects, _ = filtered_images[pair_index]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'Filtering Results - {filename}', fontsize=16)
    
    # フィルタリング前
    before_vis = create_defect_visualization(
        mask_before, 
        (large_before, small_before)
    )
    axes[0, 0].imshow(before_vis)
    axes[0, 0].set_title('Before Filtering')
    
    # フィルタリング後（ラベルなし）
    after_vis_no_labels = create_defect_visualization(
        mask_after,
        (filtered_large, filtered_small)
    )
    axes[0, 1].imshow(after_vis_no_labels)
    axes[0, 1].set_title('After Filtering (No Labels)')
    
    # フィルタリング後（ラベルあり）
    after_vis_with_labels = create_defect_visualization(
        mask_after,
        (filtered_large, filtered_small),
        defects,
        True
    )
    axes[1, 0].imshow(after_vis_with_labels)
    axes[1, 0].set_title('After Filtering (With Labels)')
    
    # 凡例用の説明
    axes[1, 1].text(0.1, 0.7, 'Gray: Processing Area', fontsize=12)
    axes[1, 1].text(0.1, 0.5, 'Red: Canny (Large Defects)', fontsize=12)
    axes[1, 1].text(0.1, 0.3, 'Blue: DoG (Small Defects)', fontsize=12)
    axes[1, 1].axis('off')
    
    for ax in axes.ravel()[:3]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# NG画像のフィルタリング結果を表示
print("NG画像のフィルタリング結果:")
if processed_ng_images and filtered_ng_images:
    visualize_filtering_results(processed_ng_images, filtered_ng_images, 0)

# OK画像のフィルタリング結果を表示
print("\nOK画像のフィルタリング結果:")
if processed_ok_images and filtered_ok_images:
    visualize_filtering_results(processed_ok_images, filtered_ok_images, 0)
```

主な改良点：
1. 関数の分割と責任の明確化
2. 詳細なdocstringsの追加
3. 特徴量計測処理の独立化
4. 可視化機能の強化
5. わかりやすい凡例の追加
6. フィルタリング前後の比較を4分割で表示

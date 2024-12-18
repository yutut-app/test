```python
# 7. 欠陥候補のフィルタリング

def compute_defect_features(region, label, detection_method):
    """
    欠陥領域の特徴量を計算します
    
    引数:
        region: 領域のプロパティ
        label (int): 欠陥のラベル番号
        detection_method (str): 検出方法（'canny'または'dog'）
    
    戻り値:
        dict: 欠陥の特徴量情報
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
    サイズに基づいて欠陥候補をフィルタリングし、特徴量を計算します
    
    引数:
        binary_image (ndarray): 二値化画像
        min_size (int): 最小サイズ
        max_size (int): 最大サイズ
        detection_method (str): 検出方法
        start_label (int): ラベル開始番号
    
    戻り値:
        list: 欠陥情報のリスト
    """
    defects = []
    labels = measure.label(binary_image, connectivity=2)
    
    for region in measure.regionprops(labels):
        if min_size <= region.area <= max_size:
            defect_info = compute_defect_features(
                region,
                start_label + len(defects) + 1,
                detection_method
            )
            defects.append(defect_info)
    
    return defects

def filter_and_measure_defects(edge_image, large_defects, small_defects, mask):
    """
    CannyとDoGそれぞれの結果に対して、適切なサイズ範囲でフィルタリングを行います
    
    引数:
        edge_image (ndarray): エッジ画像
        large_defects (ndarray): Canny検出結果
        small_defects (ndarray): DoG検出結果
        mask (ndarray): マスク画像
    
    戻り値:
        tuple: フィルタリング後の欠陥情報リストと画像
    """
    # マスクエッジの作成
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    
    # 大きな欠陥（Canny）の処理
    binary_large = (large_defects > 0).astype(np.uint8)
    binary_large[mask_edges_with_margin > 0] = 0
    large_defects_info = filter_defects_by_size(
        binary_large, 
        min_large_defect_size, 
        max_large_defect_size, 
        'canny', 
        0
    )
    
    # 小さな欠陥（DoG）の処理
    binary_small = (small_defects > 0).astype(np.uint8)
    binary_small[mask_edges_with_margin > 0] = 0
    small_defects_info = filter_defects_by_size(
        binary_small, 
        min_small_defect_size, 
        max_small_defect_size, 
        'dog', 
        len(large_defects_info)
    )
    
    # フィルタリング後の画像を作成
    filtered_large = np.zeros_like(large_defects)
    filtered_small = np.zeros_like(small_defects)
    
    for defect in large_defects_info:
        y1, x1 = defect['y'], defect['x']
        y2, x2 = y1 + defect['height'], x1 + defect['width']
        filtered_large[y1:y2, x1:x2] = binary_large[y1:y2, x1:x2]
    
    for defect in small_defects_info:
        y1, x1 = defect['y'], defect['x']
        y2, x2 = y1 + defect['height'], x1 + defect['width']
        filtered_small[y1:y2, x1:x2] = binary_small[y1:y2, x1:x2]
    
    return large_defects_info + small_defects_info, filtered_large, filtered_small

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
        defects, filtered_large, filtered_small = filter_and_measure_defects(
            mask, large_defects, small_defects, mask
        )
        filtered_images.append((mask, defects, filtered_large, filtered_small, original_filename))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images = process_images_for_filtering(completed_ng_images)
filtered_ok_images = process_images_for_filtering(completed_ok_images)
```

```python
# フィルタリング結果の可視化

def visualize_filtering_results(completed_images, filtered_images, pair_index):
    """
    フィルタリング前後の結果を可視化します
    
    引数:
        completed_images (list): フィルタリング前の画像リスト
        filtered_images (list): フィルタリング後の画像リスト
        pair_index (int): 表示するペアのインデックス
    """
    if not completed_images or not filtered_images or pair_index >= len(completed_images):
        print("指定されたインデックスの画像が存在しません")
        return
    
    # フィルタリング前の画像を取得
    mask_before, large_before, small_before, _ = completed_images[pair_index]
    
    # フィルタリング後の画像を取得
    mask_after, _, filtered_large, filtered_small, filename = filtered_images[pair_index]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Filtering Results - {filename}', fontsize=16)
    
    # フィルタリング前の結果表示
    colored_before = np.zeros((*large_before.shape, 3), dtype=np.uint8)
    colored_before[mask_before > 0] = [200, 200, 200]  # 加工領域をグレーで表示
    colored_before[large_before > 0] = [255, 0, 0]     # Canny結果を赤で表示
    colored_before[small_before > 0] = [0, 0, 255]     # DoG結果を青で表示
    axes[0].imshow(colored_before)
    axes[0].set_title('Before Filtering\nGray: Processing Area\nRed: Canny (Large), Blue: DoG (Small)')
    
    # フィルタリング後の結果表示
    colored_after = np.zeros((*filtered_large.shape, 3), dtype=np.uint8)
    colored_after[mask_after > 0] = [200, 200, 200]    # 加工領域をグレーで表示
    colored_after[filtered_large > 0] = [255, 0, 0]    # Canny結果を赤で表示
    colored_after[filtered_small > 0] = [0, 0, 255]    # DoG結果を青で表示
    axes[1].imshow(colored_after)
    axes[1].set_title('After Filtering\nGray: Processing Area\nRed: Canny (Large), Blue: DoG (Small)')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# NG画像のフィルタリング結果を表示
print("NG画像のフィルタリング結果:")
if completed_ng_images and filtered_ng_images:
    visualize_filtering_results(completed_ng_images, filtered_ng_images, 0)

# OK画像のフィルタリング結果を表示
print("\nOK画像のフィルタリング結果:")
if completed_ok_images and filtered_ok_images:
    visualize_filtering_results(completed_ok_images, filtered_ok_images, 0)
```

主な改良点：
1. 関数の責任を明確に分離
2. 詳細なdocstrings追加
3. 特徴量計算を独立した関数に
4. フィルタリング前後の画像を保持
5. 可視化機能の改善
6. 加工領域の表示追加

これにより：
- コードの保守性が向上
- 処理の流れが理解しやすく
- フィルタリングの効果が視覚的に確認可能
になっています。

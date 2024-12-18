```python
# 7. 欠陥候補のフィルタリング

def measure_region_properties(region, label, detection_method):
    """
    領域の特徴量を計測します
    
    引数:
        region: 領域のプロパティ
        label (int): 欠陥のラベル番号
        detection_method (str): 検出方法（'canny' または 'dog'）
    
    戻り値:
        dict: 欠陥の特徴量情報
    """
    y, x = region.bbox[0], region.bbox[1]
    h, w = region.bbox[2] - y, region.bbox[3] - x
    
    return {
        'label': label,
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
        'max_length': max(w, h),
        'detection_method': detection_method
    }

def process_defect_regions(binary_image, min_size, max_size, detection_method, start_label):
    """
    指定されたサイズ範囲で欠陥候補領域を処理します
    
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
            defect_info = measure_region_properties(
                region, 
                start_label + len(defects) + 1,
                detection_method
            )
            defects.append(defect_info)
            
    return defects

def filter_defects(large_defects, small_defects, mask):
    """
    Cannyエッジ検出とDoG検出の結果をフィルタリングします
    
    引数:
        large_defects (ndarray): Cannyエッジ検出結果
        small_defects (ndarray): DoG検出結果
        mask (ndarray): マスク画像
        
    戻り値:
        list: フィルタリング後の欠陥情報リスト
    """
    # マスクエッジの作成
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    
    # 大きな欠陥（Canny）の処理
    binary_large = (large_defects > 0).astype(np.uint8)
    binary_large[mask_edges_with_margin > 0] = 0
    large_defects_info = process_defect_regions(
        binary_large,
        min_large_defect_size,
        max_large_defect_size,
        'canny',
        0
    )
    
    # 小さな欠陥（DoG）の処理
    binary_small = (small_defects > 0).astype(np.uint8)
    binary_small[mask_edges_with_margin > 0] = 0
    small_defects_info = process_defect_regions(
        binary_small,
        min_small_defect_size,
        max_small_defect_size,
        'dog',
        len(large_defects_info)
    )
    
    return large_defects_info + small_defects_info

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
        # フィルタリングの実行
        defects = filter_defects(large_defects, small_defects, mask)
        filtered_images.append((mask, large_defects, small_defects, defects, original_filename))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images = process_images_for_filtering(completed_ng_images)
filtered_ok_images = process_images_for_filtering(completed_ok_images)
```

```python
# フィルタリング結果の可視化

def visualize_filtered_defects(processed_images, pair_index):
    """
    フィルタリング結果を可視化します。
    ラベルありとラベルなしの両方のバージョンを表示します。
    
    引数:
        processed_images (list): 処理済み画像のリスト
        pair_index (int): 表示するペアのインデックス
    """
    if not processed_images or pair_index >= len(processed_images):
        print("指定されたインデックスの画像が存在しません")
        return
    
    mask, large_defects, small_defects, defects, filename = processed_images[pair_index]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Filtered Defects - {filename}', fontsize=16)
    
    # 共通の背景とベース画像の作成
    for ax in axes:
        # 加工領域をグレーで表示
        colored_result = np.zeros((*large_defects.shape, 3), dtype=np.uint8)
        colored_result[mask > 0] = [200, 200, 200]
        # 欠陥候補を色分けして表示
        colored_result[large_defects > 0] = [255, 0, 0]
        colored_result[small_defects > 0] = [0, 0, 255]
        ax.imshow(colored_result)
    
    # ラベルなしの結果
    axes[0].set_title('Without Labels\nGray: Processing Area, Red: Canny (Large), Blue: DoG (Small)')
    
    # ラベルありの結果
    axes[1].set_title('With Labels\nGray: Processing Area, Red: Canny (Large), Blue: DoG (Small)')
    for defect in defects:
        color = 'red' if defect['detection_method'] == 'canny' else 'blue'
        axes[1].text(
            defect['x'], defect['y'],
            str(defect['label']),
            color=color,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# NG画像のフィルタリング結果を表示
print("NG画像のフィルタリング結果:")
if filtered_ng_images:
    visualize_filtered_defects(filtered_ng_images, 0)

# OK画像のフィルタリング結果を表示
print("\nOK画像のフィルタリング結果:")
if filtered_ok_images:
    visualize_filtered_defects(filtered_ok_images, 0)
```

主な改良点：
1. 関数の分割と責任の明確化
2. 詳細なdocstringsの追加
3. 特徴量計測の処理を独立した関数に分離
4. 可視化の改善（ラベルありなしの両方を表示）
5. 処理の流れをより明確に
6. エラーハンドリングの追加

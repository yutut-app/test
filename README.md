はい、共通部分をまとめてコードをより簡潔にします。

```python
def filter_and_measure_defects(edge_image, large_defects, small_defects, mask):
    """
    CannyとDoGそれぞれの結果に対して、適切なサイズ範囲でフィルタリングを行う
    """
    def process_regions(binary_image, min_size, max_size, detection_method, start_label):
        """
        領域の処理を行う内部関数
        """
        defects = []
        labels = measure.label(binary_image, connectivity=2)
        
        for region in measure.regionprops(labels):
            if min_size <= region.area <= max_size:
                y, x = region.bbox[0], region.bbox[1]
                h, w = region.bbox[2] - y, region.bbox[3] - x
                defect_info = {
                    'label': start_label + len(defects) + 1,
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
                defects.append(defect_info)
        return defects

    # マスクエッジの作成
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    
    # 大きな欠陥（Canny）の処理
    binary_large = (large_defects > 0).astype(np.uint8)
    binary_large[mask_edges_with_margin > 0] = 0
    large_defects = process_regions(binary_large, min_large_defect_size, max_large_defect_size, 'canny', 0)
    
    # 小さな欠陥（DoG）の処理
    binary_small = (small_defects > 0).astype(np.uint8)
    binary_small[mask_edges_with_margin > 0] = 0
    small_defects = process_regions(binary_small, min_small_defect_size, max_small_defect_size, 'dog', len(large_defects))
    
    # 全ての欠陥を統合
    return large_defects + small_defects

def process_images_for_filtering(completed_edge_images):
    filtered_images = []
    for binarized_image, edge_image, original_filename in completed_edge_images:
        # Canny+DoGの検出結果を取得
        combined_result, large_defects, small_defects = combine_defect_detection(edge_image, binarized_image)
        
        # それぞれのサイズ範囲でフィルタリング
        defects = filter_and_measure_defects(combined_result, large_defects, small_defects, binarized_image)
        
        filtered_images.append((original_filename, binarized_image, edge_image, defects))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images_label1 = process_images_for_filtering(completed_ng_images_label1)
filtered_ng_images_label2 = process_images_for_filtering(completed_ng_images_label2)
filtered_ng_images_label3 = process_images_for_filtering(completed_ng_images_label3)
filtered_ok_images = process_images_for_filtering(completed_ok_images)
```

主な変更点：
1. 共通の処理を`process_regions`という内部関数にまとめました。
2. この内部関数で欠陥情報の生成を一元化しました。
3. ラベル番号の連番を保持するため、小さな欠陥の処理時に開始ラベル番号を渡すようにしました。
4. コードの重複を削除し、より保守しやすい構造にしました。

この変更により：
- コードの重複が減少
- 処理の一貫性が向上
- 将来の修正がより容易に
- 可読性が向上

という利点が得られます。機能は完全に同じですが、より整理された形になりました。

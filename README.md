承知しました。以下に、「7. エッジの補完とラベリング処理」と「8. 欠陥候補のフィルタリング」の全体を省略せずに記載します。

```python
# 7. エッジの補完（ラベリング処理は削除）

def create_mask_edge_margin(mask, margin):
    mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
    dilated_edges = cv2.dilate(mask_edges, kernel, iterations=1)
    return dilated_edges

def complete_edges(edge_image, mask):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    skeleton = skeletonize(edge_image > 0)
    kernel = np.ones(edge_kernel_size, np.uint8)
    
    # オープン処理でノイズを削除
    opened_skeleton = cv2.morphologyEx(skeleton.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=edge_open_iterations)
    
    # クローズ処理でエッジを接続
    connected_skeleton = cv2.morphologyEx(opened_skeleton, cv2.MORPH_CLOSE, kernel, iterations=edge_close_iterations)
    
    completed_edges = np.maximum(edge_image, connected_skeleton * 255)
    completed_edges = np.where(mask_edges_with_margin > 0, edge_image, completed_edges)
    return completed_edges.astype(np.uint8)

def process_images_for_edge_completion(edged_images):
    completed_edge_images = []
    for binarized_image, edge_image, original_filename in edged_images:
        completed_edges = complete_edges(edge_image, binarized_image)
        completed_edge_images.append((binarized_image, completed_edges, original_filename))
    return completed_edge_images

# NGとOK画像に対してエッジ補完を実行
completed_ng_images_label1 = process_images_for_edge_completion(edged_ng_images_label1)
completed_ng_images_label2 = process_images_for_edge_completion(edged_ng_images_label2)
completed_ng_images_label3 = process_images_for_edge_completion(edged_ng_images_label3)
completed_ok_images = process_images_for_edge_completion(edged_ok_images)

# 8. 欠陥候補のフィルタリングと特徴量計算

def filter_and_measure_defects(edge_image, mask, min_size, max_size):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    binary_edge_image = (edge_image > 0).astype(np.uint8)
    binary_edge_image[mask_edges_with_margin > 0] = 0  # マスクエッジ部分を除外
    
    labels = measure.label(binary_edge_image, connectivity=2)
    defects = []
    
    for region in measure.regionprops(labels):
        if min_size <= region.area <= max_size:
            y, x = region.bbox[0], region.bbox[1]
            h, w = region.bbox[2] - y, region.bbox[3] - x
            defect_info = {
                'label': region.label,
                'x': x, 'y': y, 'width': w, 'height': h,
                'area': region.area,
                'centroid_y': region.centroid[0], 'centroid_x': region.centroid[1],
                'perimeter': region.perimeter,
                'eccentricity': region.eccentricity,
                'orientation': region.orientation,
                'major_axis_length': region.major_axis_length,
                'minor_axis_length': region.minor_axis_length,
                'solidity': region.solidity,
                'extent': region.extent,
                'aspect_ratio': max(w, h) / min(w, h) if min(w, h) > 0 else 0,
                'max_length': max(w, h)
            }
            defects.append(defect_info)
    
    return defects

def process_images_for_filtering(completed_edge_images):
    filtered_images = []
    for binarized_image, edge_image, original_filename in completed_edge_images:
        defects = filter_and_measure_defects(edge_image, binarized_image, min_defect_size, max_defect_size)
        filtered_images.append((original_filename, binarized_image, edge_image, defects))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images_label1 = process_images_for_filtering(completed_ng_images_label1)
filtered_ng_images_label2 = process_images_for_filtering(completed_ng_images_label2)
filtered_ng_images_label3 = process_images_for_filtering(completed_ng_images_label3)
filtered_ok_images = process_images_for_filtering(completed_ok_images)

# フィルタリング結果の可視化
def visualize_filtered_defects(image_name, image, defects, mask):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image, cmap='gray')
    
    # マスクのエッジを可視化
    mask_edges = create_mask_edge_margin(mask, mask_edge_margin)
    ax.imshow(mask_edges, alpha=0.3, cmap='cool')
    
    for defect in defects:
        rect = plt.Rectangle((defect['x'], defect['y']), defect['width'], defect['height'],
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(defect['x'], defect['y'], str(defect['label']), color='red', fontsize=12)
    
    plt.title(f"Filtered Defects with Mask Edges - {image_name}", fontsize=20)
    plt.axis('off')
    plt.show()

# フィルタリング結果の可視化（例：最初のNG画像）
if filtered_ng_images_label1:
    image_name, binarized_image, edge_image, filtered_defects = filtered_ng_images_label1[0]
    visualize_filtered_defects(image_name, edge_image, filtered_defects, binarized_image)
```

この改良版では以下の変更が行われています：

1. エッジの補完処理を保持しつつ、ラベリング処理を削除しました。
2. フィルタリングと特徴量計算を一つの処理にまとめ、メモリ使用量を削減しました。
3. `filter_and_measure_defects` 関数内で、指定されたサイズ範囲内の欠陥候補のみを処理し、特徴量を計算します。
4. 処理の流れを維持しつつ、各ステップでのメモリ使用量を削減しています。

これらの変更により、メモリエラーを回避しつつ、必要な処理を行うことができるはずです。また、可視化関数も含めているので、結果を視覚的に確認することもできます。

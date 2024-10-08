承知しました。# 7. エッジの補完とラベリング処理と# 8. 欠陥候補のフィルタリングの全てを省略せずに記載します。

```python
# 7. エッジの補完とラベリング処理

def create_mask_edge_margin(mask, margin):
    mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
    dilated_edges = cv2.dilate(mask_edges, kernel, iterations=1)
    return dilated_edges

def complete_edges(edge_image, mask):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    skeleton = skeletonize(edge_image > 0)
    kernel = np.ones(edge_kernel_size, np.uint8)
    
    opened_skeleton = cv2.morphologyEx(skeleton.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=edge_open_iterations)
    connected_skeleton = cv2.morphologyEx(opened_skeleton, cv2.MORPH_CLOSE, kernel, iterations=edge_close_iterations)
    
    completed_edges = np.maximum(edge_image, connected_skeleton * 255)
    completed_edges = np.where(mask_edges_with_margin > 0, edge_image, completed_edges)
    return completed_edges.astype(np.uint8)

def label_edges(edge_image, mask):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    binary_edge_image = (edge_image > 0).astype(np.uint8)
    binary_edge_image[mask_edges_with_margin > 0] = 0
    labels = measure.label(binary_edge_image, connectivity=2)
    return labels

def process_images_for_labeling(edged_images):
    labeled_images = []
    for binarized_image, edge_image, original_filename in edged_images:
        completed_edges = complete_edges(edge_image, binarized_image)
        labels = label_edges(completed_edges, binarized_image)
        labeled_images.append((binarized_image, completed_edges, labels, original_filename))
    return labeled_images

# NGとOK画像に対してラベリング処理を実行
labeled_ng_images_label1 = process_images_for_labeling(edged_ng_images_label1)
labeled_ng_images_label2 = process_images_for_labeling(edged_ng_images_label2)
labeled_ng_images_label3 = process_images_for_labeling(edged_ng_images_label3)
labeled_ok_images = process_images_for_labeling(edged_ok_images)

# 8. 欠陥候補のフィルタリング

def measure_defect(region):
    y, x = region.bbox[0], region.bbox[1]
    h, w = region.bbox[2] - y, region.bbox[3] - x
    return {
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

def filter_and_measure_defects(labels, min_size, max_size):
    defects = []
    for region in measure.regionprops(labels):
        if min_size <= region.area <= max_size:
            defect_info = measure_defect(region)
            defects.append(defect_info)
    return defects

def process_images_for_filtering(labeled_images):
    filtered_images = []
    for binarized_image, edge_image, labels, original_filename in labeled_images:
        filtered_defects = filter_and_measure_defects(labels, min_defect_size, max_defect_size)
        filtered_images.append((original_filename, binarized_image, edge_image, filtered_defects))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images_label1 = process_images_for_filtering(labeled_ng_images_label1)
filtered_ng_images_label2 = process_images_for_filtering(labeled_ng_images_label2)
filtered_ng_images_label3 = process_images_for_filtering(labeled_ng_images_label3)
filtered_ok_images = process_images_for_filtering(labeled_ok_images)

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

1. エッジの補完処理(`complete_edges`)とラベリング処理(`label_edges`)を分離しました。

2. `process_images_for_labeling`関数では、完成したエッジ画像とラベル画像の両方を返すようにしました。

3. フィルタリング処理では、ラベル付けされた画像を入力として受け取り、サイズフィルタリングを適用した後に必要な欠陥情報を計算します。

4. `measure_defect`関数を導入し、個々の欠陥領域の測定を行います。

5. `filter_and_measure_defects`関数では、サイズフィルタリングと測定を同時に行い、メモリ使用量を削減します。

6. 可視化関数(`visualize_filtered_defects`)は変更せずに保持しています。

これらの変更により、メモリ使用量が削減され、大規模な画像セットでも効率的に処理できるようになるはずです。また、処理の各段階がより明確に分離され、コードの可読性と保守性が向上しています。

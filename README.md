承知しました。メモリ使用量を削減し、処理を最適化するために、以下のように改良します。

まず、"4. ワーク接合部の削除"の部分を改良します：

```python
# 4. ワーク接合部の削除

def template_matching(image, template_path):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc

def remove_joint_part(image_path, keyence_image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keyence_image = cv2.imread(keyence_image_path, cv2.IMREAD_GRAYSCALE)
    
    right_val, _ = template_matching(keyence_image, right_template_path)
    left_val, _ = template_matching(keyence_image, left_template_path)
    
    if right_val > left_val:
        cropped_image = image[:, :-crop_width]
        cropped_keyence_image = keyence_image[:, :-crop_width]
    else:
        cropped_image = image[:, crop_width:]
        cropped_keyence_image = keyence_image[:, crop_width:]
    
    return cropped_image, cropped_keyence_image

def process_images(image_pairs):
    updated_images = []
    for origin_image_path, keyence_image_path, original_filename in image_pairs:
        cropped_image, cropped_keyence_image = remove_joint_part(origin_image_path, keyence_image_path)
        updated_images.append((cropped_image, cropped_keyence_image, original_filename))
    return updated_images

# NGとOK画像に対して接合部削除を実行
updated_ng_images_label1 = process_images(ng_images_label1)
updated_ng_images_label2 = process_images(ng_images_label2)
updated_ng_images_label3 = process_images(ng_images_label3)
updated_ok_images = process_images(ok_images)
```

次に、"7. エッジの補完とラベリング処理"と"8. 欠陥候補のフィルタリング"を改良します：

```python
# 7. エッジの補完とラベリング処理

def complete_edges(edge_image, mask):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    skeleton = skeletonize(edge_image > 0)
    kernel = np.ones(edge_kernel_size, np.uint8)
    
    opened_skeleton = cv2.morphologyEx(skeleton.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=edge_open_iterations)
    connected_skeleton = cv2.morphologyEx(opened_skeleton, cv2.MORPH_CLOSE, kernel, iterations=edge_close_iterations)
    
    completed_edges = np.maximum(edge_image, connected_skeleton * 255)
    completed_edges = np.where(mask_edges_with_margin > 0, edge_image, completed_edges)
    return completed_edges.astype(np.uint8)

def process_images_for_labeling(edged_images):
    labeled_images = []
    for binarized_image, edge_image, original_filename in edged_images:
        completed_edges = complete_edges(edge_image, binarized_image)
        labeled_images.append((binarized_image, completed_edges, original_filename))
    return labeled_images

# NGとOK画像に対してラベリング処理を実行
labeled_ng_images_label1 = process_images_for_labeling(edged_ng_images_label1)
labeled_ng_images_label2 = process_images_for_labeling(edged_ng_images_label2)
labeled_ng_images_label3 = process_images_for_labeling(edged_ng_images_label3)
labeled_ok_images = process_images_for_labeling(edged_ok_images)

# 8. 欠陥候補のフィルタリング

def label_and_measure_defects(edge_image, mask):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    binary_edge_image = (edge_image > 0).astype(np.uint8)
    binary_edge_image[mask_edges_with_margin > 0] = 0
    labels = measure.label(binary_edge_image, connectivity=2)
    return labels

def filter_and_measure_defects(labels, min_size, max_size):
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

def process_images_for_filtering(labeled_images):
    filtered_images = []
    for binarized_image, edge_image, original_filename in labeled_images:
        labels = label_and_measure_defects(edge_image, binarized_image)
        filtered_defects = filter_and_measure_defects(labels, min_defect_size, max_defect_size)
        filtered_images.append((original_filename, binarized_image, edge_image, filtered_defects))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images_label1 = process_images_for_filtering(labeled_ng_images_label1)
filtered_ng_images_label2 = process_images_for_filtering(labeled_ng_images_label2)
filtered_ng_images_label3 = process_images_for_filtering(labeled_ng_images_label3)
filtered_ok_images = process_images_for_filtering(labeled_ok_images)
```

これらの変更により：

1. ワーク接合部の削除処理では、グレースケールで画像を読み込むことでメモリ使用量を削減しています。

2. エッジの補完とラベリング処理では、defect_infoの計算を延期し、ラベリングのみを行います。

3. 欠陥候補のフィルタリング処理で、サイズフィルタリングを行った後に、必要な欠陥情報のみを計算します。

これらの最適化により、メモリ使用量が大幅に削減され、処理速度も向上するはずです。エラーが解消されない場合は、さらなる最適化や、より大きなメモリを持つマシンでの実行を検討する必要があるかもしれません。

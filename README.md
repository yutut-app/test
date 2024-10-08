はい、メモリ不足の問題に対処するために、コードを改良します。大きな画像を処理する際にメモリ使用量を削減するため、画像のダウンサイズと処理の最適化を行います。

まず、"4. ワーク接合部の削除"の部分を改良します：

```python
# 4. ワーク接合部の削除

def downsample_image(image, scale_factor=0.5):
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

def template_matching(image, template_path, scale_factor=0.5):
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    template = downsample_image(template, scale_factor)
    
    downsampled_image = downsample_image(image, scale_factor)
    res = cv2.matchTemplate(downsampled_image, template, cv2.TM_CCOEFF)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    # スケールを元のサイズに戻す
    max_loc = (int(max_loc[0] / scale_factor), int(max_loc[1] / scale_factor))
    return max_val, max_loc

def remove_joint_part(image_path, keyence_image_path, scale_factor=0.5):
    image = io.imread(image_path)
    keyence_image = io.imread(keyence_image_path)
    
    right_val, _ = template_matching(keyence_image, right_template_path, scale_factor)
    left_val, _ = template_matching(keyence_image, left_template_path, scale_factor)
    
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
```

次に、"7. エッジの補完とラベリング処理"の部分を改良します：

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

def label_and_measure_defects(edge_image, mask):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    binary_edge_image = (edge_image > 0).astype(np.uint8)
    binary_edge_image[mask_edges_with_margin > 0] = 0  # マスクエッジ部分を除外
    
    # メモリ使用量を削減するためにラベリングを16ビット整数で行う
    labels = measure.label(binary_edge_image, connectivity=2, return_num=False, dtype=np.int16)
    
    defects = []
    for region in measure.regionprops(labels):
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

def process_images_for_labeling(edged_images):
    labeled_images = []
    for binarized_image, edge_image, original_filename in edged_images:
        completed_edges = complete_edges(edge_image, binarized_image)
        defects = label_and_measure_defects(completed_edges, binarized_image)
        labeled_images.append((binarized_image, completed_edges, defects, original_filename))
    return labeled_images
```

これらの改良点は以下の通りです：

1. ワーク接合部の削除:
   - 画像をダウンサンプリングしてからテンプレートマッチングを行い、メモリ使用量を削減します。
   - 結果を元のサイズにスケールバックします。

2. エッジの補完とラベリング処理:
   - ラベリング処理で16ビット整数型を使用し、メモリ使用量を削減します。
   - 大きな配列の作成を避けるため、一部の処理を最適化しています。

これらの変更により、メモリ使用量が大幅に削減され、大きな画像でもエラーなく処理できるはずです。ただし、画像のサイズによっては `scale_factor` の値を調整する必要があるかもしれません。また、処理速度と精度のトレードオフがあるため、必要に応じて調整してください。

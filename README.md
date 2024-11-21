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

def label_and_measure_defects(edge_image, mask):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    binary_edge_image = (edge_image > 0).astype(np.uint8)
    binary_edge_image[mask_edges_with_margin > 0] = 0  # マスクエッジ部分を除外
    labels = measure.label(binary_edge_image, connectivity=2)
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
    for binarized_image, edge_image in edged_images:
        completed_edges = complete_edges(edge_image, binarized_image)
        defects = label_and_measure_defects(completed_edges, binarized_image)
        labeled_images.append((binarized_image, completed_edges, defects))
    return labeled_images

# NGとOK画像に対してラベリング処理を実行
labeled_ng_images_label1 = process_images_for_labeling(edged_ng_images_label1)
labeled_ng_images_label2 = process_images_for_labeling(edged_ng_images_label2)
labeled_ng_images_label3 = process_images_for_labeling(edged_ng_images_label3)
labeled_ok_images = process_images_for_labeling(edged_ok_images)

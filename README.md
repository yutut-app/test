申し訳ありません。ご指摘ありがとうございます。それでは、`binarize_images`, `process_images_for_edge_detection`, `process_images_for_labeling` 関数も含めて、全ての変更を省略せずに記載します。

```python
def binarize_images(image_pairs):
    binarized_images = []
    for cropped_image, cropped_keyence_image, original_filename in image_pairs:
        binarized_image = binarize_image(cropped_image)
        binarized_images.append((binarized_image, cropped_keyence_image, original_filename))
    return binarized_images

def process_images_for_edge_detection(binarized_images):
    edged_images = []
    for binarized_image, cropped_keyence_image, original_filename in binarized_images:
        edge_image = detect_edges_and_texture(cropped_keyence_image, binarized_image)
        edged_images.append((binarized_image, edge_image, original_filename))
    return edged_images

def process_images_for_labeling(edged_images):
    labeled_images = []
    for binarized_image, edge_image, original_filename in edged_images:
        completed_edges = complete_edges(edge_image, binarized_image)
        defects = label_and_measure_defects(completed_edges, binarized_image)
        labeled_images.append((binarized_image, completed_edges, defects, original_filename))
    return labeled_images

def process_images_for_filtering(labeled_images, image_type):
    filtered_images = []
    for binarized_image, edge_image, defects, original_filename in labeled_images:
        filtered_defects = filter_defects_by_max_length(defects, min_defect_size, max_defect_size)
        
        for j, defect in enumerate(filtered_defects, 1):
            defect['label'] = j
        
        filtered_images.append((original_filename, binarized_image, edge_image, filtered_defects))
    return filtered_images

def process_images_for_saving(filtered_images, base_output_dir, image_label):
    all_defects_data = []
    
    for original_filename, binarized_image, edge_image, defects in filtered_images:
        image_type = "ng" if image_label == 1 else "ok"
        output_dir = os.path.join(base_output_dir, image_type, original_filename.replace(".jpg", ""))
        os.makedirs(output_dir, exist_ok=True)
        
        for defect in defects:
            output_filename = save_defect_image(edge_image, defect, output_dir, original_filename, defect['label'])
            
            defect_data = {
                'image_name': original_filename,
                'defect_image': os.path.join(image_type, original_filename.replace(".jpg", ""), output_filename),
                'Image_label': image_label,
                'defect_label': 0,  # デフォルトで0（OK）とする
            }
            defect_data.update(defect)
            all_defects_data.append(defect_data)
    
    return all_defects_data
```

これらの変更により、元のファイル名（`original_filename`）が処理の全段階を通して保持され、最終的に `defect_data` の `'image_name'` フィールドに保存されます。また、出力ディレクトリ構造も元のファイル名に基づいて作成されるように変更されています。

これらの変更を適用することで、元の "Shape" 画像のファイル名が正しく保持され、最終的なCSVファイルに記録されるようになります。

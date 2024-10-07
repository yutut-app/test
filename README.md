承知しました。元の画像ファイル名を保持するために、コードを修正する必要があります。この変更を行うには、画像の読み込みから最終的なデータ保存までの処理を通して、元のファイル名を追跡する必要があります。以下に、変更が必要な箇所を示します。

まず、`load_origin_keyence_images` 関数を修正して、元のファイル名を保持するようにします：

```python
def load_origin_keyence_images(directory):
    normal_images = {}
    shape_images = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "Normal" in file and file.endswith(".jpg"):
                base_name = file.replace("Normal", "")
                normal_images[base_name] = (os.path.join(root, file), file)  # ファイル名も保存
            elif "Shape" in file and file.endswith(".jpg"):
                base_name = file.replace("Shape", "")
                shape_images[base_name] = (os.path.join(root, file), file)  # ファイル名も保存
    
    matched_images = []
    for base_name in normal_images:
        if base_name in shape_images:
            matched_images.append((normal_images[base_name][0], shape_images[base_name][0], shape_images[base_name][1]))  # Shape画像のファイル名を追加
    return matched_images
```

次に、`process_images` 関数を修正して、元のファイル名を渡すようにします：

```python
def process_images(image_pairs):
    updated_images = []
    for origin_image_path, keyence_image_path, original_filename in image_pairs:
        cropped_image, cropped_keyence_image = remove_joint_part(origin_image_path, keyence_image_path)
        updated_images.append((cropped_image, cropped_keyence_image, original_filename))
    return updated_images
```

同様に、`binarize_images`, `process_images_for_edge_detection`, `process_images_for_labeling` 関数も修正して、元のファイル名を渡し続けるようにします。

最後に、`process_images_for_filtering` 関数を以下のように変更します：

```python
def process_images_for_filtering(labeled_images, image_type):
    filtered_images = []
    for i, (binarized_image, edge_image, defects, original_filename) in enumerate(labeled_images):
        filtered_defects = filter_defects_by_max_length(defects, min_defect_size, max_defect_size)
        
        for j, defect in enumerate(filtered_defects, 1):
            defect['label'] = j
        
        image_name = f"{image_type}_{i}"
        filtered_images.append((original_filename, binarized_image, edge_image, filtered_defects))
    return filtered_images
```

そして、最終的な `process_images_for_saving` 関数を以下のように変更します：

```python
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

これらの変更により、`defect_data` の `'image_name'` フィールドには元の "Shape" 画像のファイル名が保存されるようになります。また、出力ディレクトリ構造も元のファイル名に基づいて作成されるように変更されています。

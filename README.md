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


これらの変更により、`defect_data` の `'image_name'` フィールドには元の "Shape" 画像のファイル名が保存されるようになります。また、出力ディレクトリ構造も元のファイル名に基づいて作成されるように変更されています。
```python
def process_images_for_saving(filtered_images, base_output_dir, image_label):
    all_defects_data = {}
    
    for original_filename, binarized_image, edge_image, defects in filtered_images:
        image_type = "ng" if image_label == 1 else "ok"
        output_dir = os.path.join(base_output_dir, image_type, original_filename.replace(".jpg", ""))
        os.makedirs(output_dir, exist_ok=True)
        
        defects_data = []
        for defect in defects:
            output_filename = save_defect_image(edge_image, defect, output_dir, original_filename, defect['label'])
            
            defect_data = {
                'image_name': original_filename,
                'defect_image': os.path.join(image_type, original_filename.replace(".jpg", ""), output_filename),
                'Image_label': image_label,
                'defect_label': 0,  # デフォルトで0（OK）とする
            }
            defect_data.update(defect)
            defects_data.append(defect_data)
        
        all_defects_data[original_filename] = defects_data
    
    return all_defects_data

# メインの処理部分
output_dir = os.path.join(output_data_dir, "defect_images")
os.makedirs(output_dir, exist_ok=True)

all_defects_data = {}
all_defects_data.update(process_images_for_saving(filtered_ng_images_label1, output_dir, 1))
all_defects_data.update(process_images_for_saving(filtered_ng_images_label2, output_dir, 1))
all_defects_data.update(process_images_for_saving(filtered_ng_images_label3, output_dir, 1))
all_defects_data.update(process_images_for_saving(filtered_ok_images, output_dir, 0))

# CSVファイルに出力
for original_filename, defects_data in all_defects_data.items():
    df = pd.DataFrame(defects_data)
    csv_output_dir = os.path.join(output_data_dir, "defect_data", original_filename.replace(".jpg", ""))
    os.makedirs(csv_output_dir, exist_ok=True)
    csv_output_path = os.path.join(csv_output_dir, "defects_data.csv")
    df.to_csv(csv_output_path, index=False)
    print(f"Defects data for {original_filename} saved to {csv_output_path}")
```

これらの変更により、元のファイル名（`original_filename`）が処理の全段階を通して保持され、最終的に `defect_data` の `'image_name'` フィールドに保存されます。また、出力ディレクトリ構造も元のファイル名に基づいて作成されるように変更されています。

これらの変更を適用することで、元の "Shape" 画像のファイル名が正しく保持され、最終的なCSVファイルに記録されるようになります。

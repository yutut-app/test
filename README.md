欠陥部分を四角形で切り出すようにし、元の `shape_images` からも対応する画像を切り出すようにコードを変更しました。また、各画像を適切なディレクトリに保存するように修正しました。

### 改良したコード

#### 9. 欠陥候補の画像の保存とCSV出力（改良版）

```python
def save_defect_image(image, defect, output_dir, image_name, defect_number, is_binarized=True):
    x1, y1, x2, y2 = defect['x'], defect['y'], defect['x'] + defect['width'], defect['y'] + defect['height']
    
    # 四角形領域を元に画像を切り出し
    defect_image = image[y1:y2, x1:x2]
    
    if is_binarized:
        enlarged_image = cv2.resize(defect_image, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    else:
        enlarged_image = defect_image
    
    # ファイル名を設定し保存
    output_filename = f"defect_{defect_number}.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, enlarged_image)
    
    return output_filename

def process_images_for_saving(filtered_images, base_output_dir, image_label, original_shape_images):
    all_defects_data = []
    
    for i, (image_name, binarized_image, edge_image, defects) in enumerate(filtered_images):
        image_type = image_name.split('_')[0]
        
        # 元の形状画像 (Shape Image) の読み込み
        original_image_path = original_shape_images[i]
        original_image = io.imread(original_image_path)
        
        # 二値化画像と元の形状画像に対して欠陥部分を切り出し保存
        output_dir_bin = os.path.join(base_output_dir, image_type, image_name, "binarized")
        output_dir_orig = os.path.join(base_output_dir, image_type, image_name, "original")
        os.makedirs(output_dir_bin, exist_ok=True)
        os.makedirs(output_dir_orig, exist_ok=True)
        
        for j, defect in enumerate(defects, 1):
            # 二値化画像からの切り出し
            bin_output_filename = save_defect_image(binarized_image, defect, output_dir_bin, image_name, defect['label'], is_binarized=True)
            
            # 元の形状画像からの切り出し
            orig_output_filename = save_defect_image(original_image, defect, output_dir_orig, image_name, defect['label'], is_binarized=False)
            
            defect_data = {
                'image_name': image_name,
                'defect_image_bin': os.path.join(image_type, image_name, "binarized", bin_output_filename),
                'defect_image_orig': os.path.join(image_type, image_name, "original", orig_output_filename),
                'Image_label': image_label,
                'defect_label': 0,  # デフォルトで0（OK）とする
            }
            defect_data.update(defect)
            all_defects_data.append(defect_data)
    
    return all_defects_data

# 元のshape_imagesディレクトリに対応する画像データを使用して欠陥候補を保存
output_dir = os.path.join(output_data_dir, "defect_images")
os.makedirs(output_dir, exist_ok=True)

all_defects_data = []
all_defects_data.extend(process_images_for_saving(filtered_ng_images_label1, output_dir, 1, [img[1] for img in ng_images_label1]))
all_defects_data.extend(process_images_for_saving(filtered_ng_images_label2, output_dir, 1, [img[1] for img in ng_images_label2]))
all_defects_data.extend(process_images_for_saving(filtered_ng_images_label3, output_dir, 1, [img[1] for img in ng_images_label3]))
all_defects_data.extend(process_images_for_saving(filtered_ok_images, output_dir, 0, [img[1] for img in ok_images]))

# CSVファイルに出力
df = pd.DataFrame(all_defects_data)
csv_output_path = os.path.join(output_data_dir, "defects_data.csv")
df.to_csv(csv_output_path, index=False)

print(f"Defects data saved to {csv_output_path}")
```

### 改良点

1. **欠陥部分を四角形で切り出し**: 元々は正方形で欠陥部分を切り出していましたが、欠陥の幅と高さに応じて正確な四角形で切り出すように変更しました。これにより、他の欠陥部分が含まれることを防ぎます。

2. **元の `shape_images` からも切り出し**: 二値化された画像だけでなく、元々の形状画像（`shape_images`）からも対応する欠陥部分を切り出す処理を追加しました。

3. **ディレクトリ分け**: 二値化画像と元の画像を別々のディレクトリ（`binarized` と `original`）に保存し、それぞれの欠陥候補の画像を整理しました。

4. **CSV出力の拡張**: CSVファイルには、各欠陥候補に対応する二値化画像と元画像のファイルパスの両方を記録しています。

これにより、欠陥部分を正確に切り出し、元の形状画像からも対応する部分を切り出して保存できるようになっています。

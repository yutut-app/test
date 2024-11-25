```python
# 9. 欠陥候補の画像の保存とCSV出力

def save_defect_image(original_image, edge_image, defect, output_dir, output_dir_original, image_name, defect_number):
    """
    欠陥候補のエッジ画像とオリジナル画像を保存
    """
    cx, cy = defect['centroid_x'], defect['centroid_y']
    size = max(defect['width'], defect['height'])
    
    x1 = max(int(cx - size), 0)
    y1 = max(int(cy - size), 0)
    x2 = min(int(cx + size), edge_image.shape[1])
    y2 = min(int(cy + size), edge_image.shape[0])
    
    # エッジ画像の切り出しと保存
    defect_edge_image = edge_image[y1:y2, x1:x2]
    enlarged_edge_image = cv2.resize(defect_edge_image, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    
    output_filename = f"defect_{defect_number}.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, enlarged_edge_image)
    
    # オリジナル画像の切り出しと保存
    defect_original_image = original_image[y1:y2, x1:x2]
    enlarged_original_image = cv2.resize(defect_original_image, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    
    output_original_filename = f"defect_{defect_number}.png"
    output_original_path = os.path.join(output_dir_original, output_original_filename)
    cv2.imwrite(output_original_path, enlarged_original_image)
    
    return output_filename

def process_images_for_saving(filtered_images, base_output_dir, image_label):
    if filtered_images is None:
        return []  # フィルタリングされた画像がない場合は空のリストを返す
    
    defects_data = []
    
    for original_filename, binarized_image, edge_image, defects in filtered_images:
        image_type = "ng" if image_label == 1 else "ok"
        
        # エッジ画像用のディレクトリ
        output_dir = os.path.join(base_output_dir, "edge_images", image_type, original_filename.replace(".jpg", ""))
        os.makedirs(output_dir, exist_ok=True)
        
        # オリジナル画像用のディレクトリ
        output_dir_original = os.path.join(base_output_dir, "original_images", image_type, original_filename.replace(".jpg", ""))
        os.makedirs(output_dir_original, exist_ok=True)
        
        for defect in defects:
            output_filename = save_defect_image(
                binarized_image, 
                edge_image, 
                defect, 
                output_dir, 
                output_dir_original,
                original_filename, 
                defect['label']
            )
            
            defect_data = {
                'image_name': original_filename,
                'defect_image': os.path.join("edge_images", image_type, original_filename.replace(".jpg", ""), output_filename),
                'original_defect_image': os.path.join("original_images", image_type, original_filename.replace(".jpg", ""), output_filename),
                'Image_label': image_label,
                'defect_label': 0,  # デフォルトで0（OK）とする
                'detection_method': defect['detection_method']
            }
            defect_data.update(defect)
            defects_data.append(defect_data)
    
    return defects_data

# メインの処理部分
output_dir = os.path.join(output_data_dir, "defect_images")
os.makedirs(output_dir, exist_ok=True)

all_defects_data = []
for name, filtered_images, label in [
    ("NG Label 1", filtered_ng_images_label1, 1),
    ("NG Label 2", filtered_ng_images_label2, 1),
    ("NG Label 3", filtered_ng_images_label3, 1),
    ("OK", filtered_ok_images, 0)
]:
    if filtered_images:
        print(f"Processing {name} images...")
        all_defects_data.extend(process_images_for_saving(filtered_images, output_dir, label))
    else:
        print(f"No data for {name} images...")

# CSVファイルに出力
csv_output_dir = os.path.join(output_data_dir, "defect_data")
os.makedirs(csv_output_dir, exist_ok=True)
csv_output_path = os.path.join(csv_output_dir, "defects_data.csv")

df = pd.DataFrame(all_defects_data)

if os.path.exists(csv_output_path):
    # 既存のCSVファイルが存在する場合、列名なしで上書き
    df.to_csv(csv_output_path, mode='a', header=False, index=False)
    print(f"Appended defects data to existing file: {csv_output_path}")
else:
    # 新規作成の場合、列名ありで保存
    df.to_csv(csv_output_path, index=False)
    print(f"Created new defects data file: {csv_output_path}")

print(f"Defects data saved to {csv_output_path}")
```

主な変更点：
1. 保存ディレクトリを "edge_images" と "original_images" に分割
2. save_defect_image 関数を修正して両方の画像を保存
3. CSVファイルに original_defect_image カラムを追加

出力ディレクトリ構造：
```
output_data_dir/
└── defect_images/
    ├── edge_images/
    │   ├── ng/
    │   │   └── [image_name]/
    │   │       └── defect_[number].png
    │   └── ok/
    │       └── [image_name]/
    │           └── defect_[number].png
    └── original_images/
        ├── ng/
        │   └── [image_name]/
        │       └── defect_[number].png
        └── ok/
            └── [image_name]/
                └── defect_[number].png
```

これにより、エッジ検出画像とオリジナル画像の両方が別々のディレクトリに保存され、CSVファイルから両方の画像への参照が可能になります。

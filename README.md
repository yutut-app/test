```python
# 9. 欠陥候補の画像の保存とCSV出力

def save_defect_image(edge_image, original_image, defect, output_dir, edge_output_dir, original_output_dir, defect_number):
    """
    欠陥部分の画像を切り出して保存（エッジ画像とオリジナル画像の両方）
    """
    cx, cy = defect['centroid_x'], defect['centroid_y']
    size = max(defect['width'], defect['height'])
    
    x1 = max(int(cx - size), 0)
    y1 = max(int(cy - size), 0)
    x2 = min(int(cx + size), edge_image.shape[1])
    y2 = min(int(cy + size), edge_image.shape[0])
    
    # エッジ画像の切り出しと保存
    edge_defect_image = edge_image[y1:y2, x1:x2]
    edge_enlarged_image = cv2.resize(edge_defect_image, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    edge_output_filename = f"defect_{defect_number}.png"
    edge_output_path = os.path.join(edge_output_dir, edge_output_filename)
    cv2.imwrite(edge_output_path, edge_enlarged_image)
    
    # オリジナル画像の切り出しと保存
    original_defect_image = original_image[y1:y2, x1:x2]
    original_enlarged_image = cv2.resize(original_defect_image, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    original_output_filename = f"defect_{defect_number}.png"
    original_output_path = os.path.join(original_output_dir, original_output_filename)
    cv2.imwrite(original_output_path, original_enlarged_image)
    
    return edge_output_filename, original_output_filename

def process_images_for_saving(filtered_images, base_output_dir, image_label):
    """
    検出結果を保存
    """
    if filtered_images is None:
        return []  # フィルタリングされた画像がない場合は空のリストを返す
    
    defects_data = []
    
    for original_filename, binarized_image, edge_image, original_image, defects in filtered_images:
        image_type = "ng" if image_label == 1 else "ok"
        base_dir = os.path.join(base_output_dir, image_type, original_filename.replace(".jpg", ""))
        
        # エッジ画像用とオリジナル画像用のディレクトリを作成
        edge_output_dir = os.path.join(base_dir, "edge")
        original_output_dir = os.path.join(base_dir, "original")
        os.makedirs(edge_output_dir, exist_ok=True)
        os.makedirs(original_output_dir, exist_ok=True)
        
        for defect in defects:
            edge_output_filename, original_output_filename = save_defect_image(
                edge_image, original_image, defect, base_dir,
                edge_output_dir, original_output_dir, defect['label']
            )
            
            defect_data = {
                'image_name': original_filename,
                'edge_defect_image': os.path.join(image_type, original_filename.replace(".jpg", ""), "edge", edge_output_filename),
                'original_defect_image': os.path.join(image_type, original_filename.replace(".jpg", ""), "original", original_output_filename),
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
        print(f"No data for {name} images.")

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
1. save_defect_image関数を改良：
   - エッジ画像とオリジナル画像の両方を切り出して保存
   - 別々のディレクトリに保存

2. process_images_for_saving関数の更新：
   - edge用とoriginal用のサブディレクトリを作成
   - 両方の画像のパスをCSVに記録

3. ディレクトリ構造の変更：
```
output_dir/
  ├── ng/
  │   └── [image_name]/
  │       ├── edge/
  │       │   └── defect_[number].png
  │       └── original/
  │           └── defect_[number].png
  └── ok/
      └── [image_name]/
          ├── edge/
          │   └── defect_[number].png
          └── original/
              └── defect_[number].png
```

4. CSVファイルの出力内容の更新：
   - edge_defect_imageとoriginal_defect_imageの両方のパスを記録

この変更により、欠陥候補領域のエッジ画像とオリジナル画像の両方を保存し、それぞれの画像への参照をCSVファイルに記録することができます。

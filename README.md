```python
# 9. 欠陥候補の画像の保存とCSV出力

def save_defect_image(edge_image, original_image, defect, output_dir, original_output_dir, image_name, defect_number):
    """
    欠陥領域の画像を保存（エッジ画像とオリジナル画像の両方）
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
    edge_filename = f"defect_{defect_number}.png"
    edge_path = os.path.join(output_dir, edge_filename)
    cv2.imwrite(edge_path, enlarged_edge_image)
    
    # オリジナル画像の切り出しと保存
    defect_original_image = original_image[y1:y2, x1:x2]
    enlarged_original_image = cv2.resize(defect_original_image, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    original_filename = f"defect_{defect_number}.png"
    original_path = os.path.join(original_output_dir, original_filename)
    cv2.imwrite(original_path, enlarged_original_image)
    
    return edge_filename, original_filename

def process_images_for_saving(filtered_images, binarized_images, base_output_dir, image_label):
    """
    検出結果を保存（エッジ画像とオリジナル画像の両方）
    """
    if filtered_images is None:
        return []
    
    defects_data = []
    
    # binarized_imagesからオリジナル画像を取得するための辞書を作成
    original_images = {original_filename: cropped_keyence_image 
                      for binarized_image, cropped_keyence_image, original_filename in binarized_images}
    
    for original_filename, binarized_image, edge_image, defects in filtered_images:
        image_type = "ng" if image_label == 1 else "ok"
        base_name = original_filename.replace(".jpg", "")
        
        # エッジ画像用のディレクトリ
        edge_output_dir = os.path.join(base_output_dir, "edge", image_type, base_name)
        os.makedirs(edge_output_dir, exist_ok=True)
        
        # オリジナル画像用のディレクトリ
        original_output_dir = os.path.join(base_output_dir, "original", image_type, base_name)
        os.makedirs(original_output_dir, exist_ok=True)
        
        # 対応するオリジナル画像を取得
        original_image = original_images.get(original_filename)
        if original_image is None:
            continue
        
        for defect in defects:
            edge_filename, original_filename = save_defect_image(
                edge_image, original_image, defect, 
                edge_output_dir, original_output_dir,
                original_filename, defect['label']
            )
            
            defect_data = {
                'image_name': original_filename,
                'edge_image': os.path.join("edge", image_type, base_name, edge_filename),
                'original_image': os.path.join("original", image_type, base_name, original_filename),
                'Image_label': image_label,
                'defect_label': 0,  # デフォルトで0（OK）とする
            }
            defect_data.update(defect)
            defects_data.append(defect_data)
    
    return defects_data

# メインの処理部分
output_dir = os.path.join(output_data_dir, "defect_images")
os.makedirs(output_dir, exist_ok=True)

all_defects_data = []
for name, filtered_images, binarized_images, label in [
    ("NG Label 1", filtered_ng_images_label1, binarized_ng_images_label1, 1),
    ("NG Label 2", filtered_ng_images_label2, binarized_ng_images_label2, 1),
    ("NG Label 3", filtered_ng_images_label3, binarized_ng_images_label3, 1),
    ("OK", filtered_ok_images, binarized_ok_images, 0)
]:
    if filtered_images:
        print(f"Processing {name} images...")
        all_defects_data.extend(process_images_for_saving(filtered_images, binarized_images, output_dir, label))
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
1. 保存先を"edge"と"original"フォルダに分離
2. save_defect_image関数を拡張し、エッジ画像とオリジナル画像の両方を保存
3. process_images_for_savingにbinarized_imagesを追加し、オリジナル画像へのアクセスを可能に
4. CSVファイルにedge_imageとoriginal_imageの両方のパスを記録

出力ディレクトリ構造：
```
output_data_dir/
    defect_images/
        edge/
            ng/
                image1/
                    defect_1.png
                    defect_2.png
                    ...
            ok/
                ...
        original/
            ng/
                image1/
                    defect_1.png
                    defect_2.png
                    ...
            ok/
                ...
    defect_data/
        defects_data.csv
```

これにより、各欠陥候補について、エッジ検出後の画像とオリジナル画像の両方が保存され、それぞれのパスがCSVファイルに記録されます。

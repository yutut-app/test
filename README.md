`# 9. 欠陥候補の画像の保存とCSV出力`を修正して、オリジナル画像とエッジ検出画像の両方を保存できるようにします。

```python
# 9. 欠陥候補の画像の保存とCSV出力

def save_defect_image(edge_image, original_image, defect, output_dir_edge, output_dir_original, image_name, defect_number):
    """
    欠陥候補領域の画像を保存（エッジ検出画像とオリジナル画像の両方）
    """
    cx, cy = defect['centroid_x'], defect['centroid_y']
    size = max(defect['width'], defect['height'])
    
    x1 = max(int(cx - size), 0)
    y1 = max(int(cy - size), 0)
    x2 = min(int(cx + size), edge_image.shape[1])
    y2 = min(int(cy + size), edge_image.shape[0])
    
    # エッジ検出画像の切り出しと保存
    defect_edge = edge_image[y1:y2, x1:x2]
    enlarged_edge = cv2.resize(defect_edge, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    edge_filename = f"defect_{defect_number}.png"
    edge_path = os.path.join(output_dir_edge, edge_filename)
    cv2.imwrite(edge_path, enlarged_edge)
    
    # オリジナル画像の切り出しと保存
    defect_original = original_image[y1:y2, x1:x2]
    enlarged_original = cv2.resize(defect_original, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    original_filename = f"defect_{defect_number}.png"
    original_path = os.path.join(output_dir_original, original_filename)
    cv2.imwrite(original_path, enlarged_original)
    
    return edge_filename, original_filename

def get_original_image(binarized_images, target_filename):
    """
    ファイル名に対応するオリジナル画像を取得
    """
    for binarized_image, cropped_keyence_image, original_filename in binarized_images:
        if original_filename == target_filename:
            return cropped_keyence_image
    return None

def process_images_for_saving(filtered_images, binarized_images, base_output_dir, image_label):
    if filtered_images is None:
        return []  # フィルタリングされた画像がない場合は空のリストを返す
    
    defects_data = []
    
    for original_filename, binarized_image, edge_image, defects in filtered_images:
        image_type = "ng" if image_label == 1 else "ok"
        base_dir = os.path.join(base_output_dir, image_type, original_filename.replace(".jpg", ""))
        
        # エッジ検出画像用とオリジナル画像用の別々のディレクトリを作成
        output_dir_edge = os.path.join(base_dir, "edge")
        output_dir_original = os.path.join(base_dir, "original")
        os.makedirs(output_dir_edge, exist_ok=True)
        os.makedirs(output_dir_original, exist_ok=True)
        
        # 対応するオリジナル画像を取得
        original_image = get_original_image(binarized_images, original_filename)
        
        if original_image is not None:
            for defect in defects:
                edge_filename, original_filename = save_defect_image(
                    edge_image, original_image, defect,
                    output_dir_edge, output_dir_original,
                    original_filename, defect['label']
                )
                
                defect_data = {
                    'image_name': original_filename,
                    'defect_image_edge': os.path.join(image_type, original_filename.replace(".jpg", ""), "edge", edge_filename),
                    'defect_image_original': os.path.join(image_type, original_filename.replace(".jpg", ""), "original", original_filename),
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
1. save_defect_imageを修正して、エッジ画像とオリジナル画像の両方を保存
2. 出力ディレクトリを"edge"と"original"に分割
3. get_original_image関数を追加して、binarized_imagesからオリジナル画像を取得
4. CSVファイルに両方の画像パスを記録するように変更
5. メイン処理でbinarized_imagesも渡すように修正

ディレクトリ構造は以下のようになります：
```
defect_images/
    ng/
        image_name/
            edge/
                defect_1.png
                defect_2.png
                ...
            original/
                defect_1.png
                defect_2.png
                ...
    ok/
        ...
```

これにより、各欠陥候補について、エッジ検出画像とオリジナル画像の両方が保存され、それぞれのパスがCSVファイルに記録されます。

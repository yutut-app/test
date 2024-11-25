# 9. 欠陥候補の画像の保存とCSV出力を修正し、バイナリイメージとオリジナル画像の両方を保存するように変更します。

```python
def save_defect_image(binary_image, original_image, defect, base_output_dir, image_type, original_filename, defect_number):
    """
    欠陥候補のバイナリイメージとオリジナル画像を保存
    """
    cx, cy = defect['centroid_x'], defect['centroid_y']
    size = max(defect['width'], defect['height'])
    
    x1 = max(int(cx - size), 0)
    y1 = max(int(cy - size), 0)
    x2 = min(int(cx + size), binary_image.shape[1])
    y2 = min(int(cy + size), binary_image.shape[0])
    
    # バイナリイメージの切り出しと保存
    binary_defect = binary_image[y1:y2, x1:x2]
    enlarged_binary = cv2.resize(binary_defect, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    
    binary_output_dir = os.path.join(base_output_dir, "binary_images", image_type, original_filename.replace(".jpg", ""))
    os.makedirs(binary_output_dir, exist_ok=True)
    
    binary_filename = f"defect_{defect_number}.png"
    binary_path = os.path.join(binary_output_dir, binary_filename)
    cv2.imwrite(binary_path, enlarged_binary)
    
    # オリジナル画像の切り出しと保存
    original_defect = original_image[y1:y2, x1:x2]
    enlarged_original = cv2.resize(original_defect, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    
    original_output_dir = os.path.join(base_output_dir, "original_images", image_type, original_filename.replace(".jpg", ""))
    os.makedirs(original_output_dir, exist_ok=True)
    
    original_filename = f"defect_{defect_number}.png"
    original_path = os.path.join(original_output_dir, original_filename)
    cv2.imwrite(original_path, enlarged_original)
    
    return binary_filename, original_filename

def process_images_for_saving(filtered_images, base_output_dir, image_label):
    if filtered_images is None:
        return []  # フィルタリングされた画像がない場合は空のリストを返す
    
    defects_data = []
    
    for original_filename, binarized_image, edge_image, defects in filtered_images:
        image_type = "ng" if image_label == 1 else "ok"
        
        # 元のKeynce画像を読み込み
        original_image_path = os.path.join(input_data_dir, 
                                         "NG" if image_label == 1 else "OK",
                                         original_filename)
        original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        
        for defect in defects:
            # バイナリとオリジナル両方の画像を保存
            binary_filename, original_filename = save_defect_image(
                edge_image, original_image, defect, base_output_dir, 
                image_type, original_filename, defect['label']
            )
            
            defect_data = {
                'image_name': original_filename,
                'binary_image': os.path.join(image_type, "binary_images", original_filename.replace(".jpg", ""), binary_filename),
                'original_image': os.path.join(image_type, "original_images", original_filename.replace(".jpg", ""), original_filename),
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
1. save_defect_image関数を拡張
   - バイナリイメージとオリジナル画像の両方を保存
   - それぞれ別のディレクトリに保存
   - 両方の画像パスを返す

2. ディレクトリ構造の変更
```
defect_images/
├── binary_images/
│   ├── ng/
│   │   └── [original_filename]/
│   │       └── defect_[number].png
│   └── ok/
│       └── [original_filename]/
│           └── defect_[number].png
└── original_images/
    ├── ng/
    │   └── [original_filename]/
    │       └── defect_[number].png
    └── ok/
        └── [original_filename]/
            └── defect_[number].png
```

3. CSVファイルの内容を拡張
   - binary_imageパス
   - original_imageパス
   両方のパスを記録

4. process_images_for_saving関数の拡張
   - オリジナル画像の読み込みを追加
   - 両方の画像を保存するように変更
   - CSVデータに両方の画像パスを追加

これにより、欠陥候補領域のバイナリイメージとオリジナル画像の両方が保存され、それぞれの画像に簡単にアクセスできるようになります。

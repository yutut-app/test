はい、CSVファイルのカラムの名前と内容を修正します。`process_images_for_saving`関数の該当部分を修正します：

```python
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
                edge_filename, original_filename_cut = save_defect_image(
                    edge_image, original_image, defect,
                    output_dir_edge, output_dir_original,
                    original_filename, defect['label']
                )
                
                defect_data = {
                    'image_name': original_filename,  # 元の画像ファイル名
                    'defect_image_name': f"defect_{defect['label']}.png",  # 切り出した画像の名前
                    'defect_image_edge_path': os.path.join(image_type, original_filename.replace(".jpg", ""), "edge", edge_filename),  # エッジ画像のパス
                    'defect_image_original_path': os.path.join(image_type, original_filename.replace(".jpg", ""), "original", original_filename_cut),  # オリジナル画像のパス
                    'Image_label': image_label,
                    'defect_label': 0,  # デフォルトで0（OK）とする
                    'detection_method': defect['detection_method']
                }
                defect_data.update(defect)
                defects_data.append(defect_data)
    
    return defects_data
```

主な変更点：
1. 'image_name'を元の画像ファイル名（original_filename）に設定
2. 'defect_image_name'として切り出した画像のファイル名（"defect_X.png"）を追加
3. パスを示すカラム名を'defect_image_edge_path'と'defect_image_original_path'に変更

これにより、CSVファイルには以下のような情報が記録されます：
- image_name: 元の画像ファイル名（例："Shape_001.jpg"）
- defect_image_name: 切り出した画像の名前（例："defect_1.png"）
- defect_image_edge_path: エッジ画像の保存パス
- defect_image_original_path: オリジナル画像の保存パス

この形式により、元画像と切り出し画像の対応関係がより明確になります。

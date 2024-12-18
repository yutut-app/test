```python
# 8. 欠陥候補の画像とCSV保存

def extract_defect_region(image, defect):
    """
    欠陥候補の領域を切り出します
    
    引数:
        image (ndarray): 元画像
        defect (dict): 欠陥の情報
        
    戻り値:
        ndarray: 切り出した欠陥領域
    """
    x = defect['x']
    y = defect['y']
    width = defect['width']
    height = defect['height']
    
    # 余白を1ピクセル追加（画像範囲内に収める）
    x1 = max(x - 1, 0)
    y1 = max(y - 1, 0)
    x2 = min(x + width + 1, image.shape[1])
    y2 = min(y + height + 1, image.shape[0])
    
    return image[y1:y2, x1:x2]

def save_defect_image(edge_image, shape_image, defect, output_dir_edge, output_dir_shape, image_name, defect_number):
    """
    欠陥候補の画像を保存します
    
    引数:
        edge_image (ndarray): エッジ画像
        shape_image (ndarray): Shape画像
        defect (dict): 欠陥情報
        output_dir_edge (str): エッジ画像の保存ディレクトリ
        output_dir_shape (str): Shape画像の保存ディレクトリ
        image_name (str): 画像名
        defect_number (int): 欠陥番号
    
    戻り値:
        tuple: 保存したエッジ画像とShape画像のファイル名
    """
    # 欠陥領域の切り出し
    defect_edge = extract_defect_region(edge_image, defect)
    defect_shape = extract_defect_region(shape_image, defect)
    
    # 必要に応じて拡大
    if enlargement_factor > 1:
        defect_edge = cv2.resize(defect_edge, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
        defect_shape = cv2.resize(defect_shape, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    
    # ファイル名の生成と保存
    filename = f"defect_{defect_number}.png"
    cv2.imwrite(os.path.join(output_dir_edge, filename), defect_edge)
    cv2.imwrite(os.path.join(output_dir_shape, filename), defect_shape)
    
    return filename, filename

def create_defect_data_entry(defect, original_filename, image_type, edge_filename, shape_filename, base_path, image_label):
    """
    欠陥データのエントリを作成します
    
    引数:
        defect (dict): 欠陥情報
        original_filename (str): 元画像のファイル名
        image_type (str): 画像タイプ（'ng'または'ok'）
        edge_filename (str): エッジ画像のファイル名
        shape_filename (str): Shape画像のファイル名
        base_path (str): ベースパス
        image_label (int): 画像ラベル
        
    戻り値:
        dict: 欠陥データのエントリ
    """
    relative_dir = os.path.join(image_type, original_filename.replace(".jpg", ""))
    
    defect_data = {
        'image_name': original_filename,
        'defect_image_name': f"defect_{defect['label']}.png",
        'defect_image_edge_path': os.path.join(relative_dir, "edge", edge_filename),
        'defect_image_shape_path': os.path.join(relative_dir, "shape", shape_filename),
        'Image_label': image_label,
        'defect_label': 0,  # デフォルトで0（OK）
        'detection_method': defect['detection_method']
    }
    defect_data.update(defect)
    
    return defect_data

def process_images_for_saving(filtered_images, base_output_dir, image_label):
    """
    画像群の保存処理を実行します
    
    引数:
        filtered_images (list): フィルタリング済み画像のリスト
        base_output_dir (str): 出力ベースディレクトリ
        image_label (int): 画像ラベル
        
    戻り値:
        list: 欠陥データのリスト
    """
    if not filtered_images:
        return []
    
    defects_data = []
    
    for mask, large_defects, small_defects, defects, original_filename in filtered_images:
        image_type = "ng" if image_label == 1 else "ok"
        base_dir = os.path.join(base_output_dir, image_type, original_filename.replace(".jpg", ""))
        
        # 保存ディレクトリの作成
        output_dir_edge = os.path.join(base_dir, "edge")
        output_dir_shape = os.path.join(base_dir, "shape")
        os.makedirs(output_dir_edge, exist_ok=True)
        os.makedirs(output_dir_shape, exist_ok=True)
        
        # 各欠陥の処理
        for defect in defects:
            # 画像の保存
            edge_filename, shape_filename = save_defect_image(
                large_defects if defect['detection_method'] == 'canny' else small_defects,
                mask,
                defect,
                output_dir_edge,
                output_dir_shape,
                original_filename,
                defect['label']
            )
            
            # データエントリの作成
            defect_data = create_defect_data_entry(
                defect,
                original_filename,
                image_type,
                edge_filename,
                shape_filename,
                base_output_dir,
                image_label
            )
            
            defects_data.append(defect_data)
    
    return defects_data

# メインの処理実行
output_dir = os.path.join(output_data_dir, "defect_images")
os.makedirs(output_dir, exist_ok=True)

# 欠陥データの収集
all_defects_data = []
for name, filtered_images, label in [
    ("NG Label 1", filtered_ng_images, 1),
    ("OK", filtered_ok_images, 0)
]:
    if filtered_images:
        print(f"Processing {name} images...")
        all_defects_data.extend(process_images_for_saving(filtered_images, output_dir, label))
    else:
        print(f"No data for {name} images.")

# CSVファイルの出力
csv_output_dir = os.path.join(output_data_dir, "defect_data")
os.makedirs(csv_output_dir, exist_ok=True)
csv_output_path = os.path.join(csv_output_dir, "defects_data.csv")

df = pd.DataFrame(all_defects_data)

if os.path.exists(csv_output_path):
    # 既存ファイルへの追記（ヘッダーなし）
    df.to_csv(csv_output_path, mode='a', header=False, index=False)
    print(f"Appended defects data to existing file: {csv_output_path}")
else:
    # 新規ファイルの作成（ヘッダーあり）
    df.to_csv(csv_output_path, index=False)
    print(f"Created new defects data file: {csv_output_path}")

print(f"Defects data saved to {csv_output_path}")
```

主な改良点：
1. 関数の分割と責任の明確化
2. 詳細なdocstringsの追加
3. 共通処理の関数化
4. 変数名の明確化
5. エラーハンドリングの改善
6. コードの構造化とモジュール化
7. 冗長な処理の削減

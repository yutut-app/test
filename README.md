```python
# 8. 欠陥候補の画像の保存とCSV出力

def extract_roi(image, defect, margin=1):
    """
    画像から欠陥領域を切り出します
    
    引数:
        image (numpy.ndarray): 入力画像
        defect (dict): 欠陥情報
        margin (int): 余白のピクセル数
        
    戻り値:
        numpy.ndarray: 切り出された領域画像
    """
    # 切り出し範囲の計算
    x1 = max(defect['x'] - margin, 0)
    y1 = max(defect['y'] - margin, 0)
    x2 = min(defect['x'] + defect['width'] + margin, image.shape[1])
    y2 = min(defect['y'] + defect['height'] + margin, image.shape[0])
    
    # 領域の切り出し
    return image[y1:y2, x1:x2]

def save_defect_roi(image, defect, output_dir, defect_number):
    """
    欠陥領域を保存します
    
    引数:
        image (numpy.ndarray): 入力画像
        defect (dict): 欠陥情報
        output_dir (str): 出力ディレクトリ
        defect_number (int): 欠陥番号
        
    戻り値:
        str: 保存されたファイル名
    """
    # 領域の切り出し
    roi = extract_roi(image, defect)
    
    # 画像の拡大
    if enlargement_factor > 1:
        roi = cv2.resize(roi, (0, 0), 
                        fx=enlargement_factor, 
                        fy=enlargement_factor)
    
    # ファイルの保存
    filename = f"defect_{defect_number}.png"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, roi)
    
    return filename

def create_defect_info(defect, image_name, defect_number, image_type, image_label):
    """
    欠陥情報辞書を作成します
    
    引数:
        defect (dict): 検出された欠陥の情報
        image_name (str): 元画像の名前
        defect_number (int): 欠陥番号
        image_type (str): 画像タイプ（'ng'または'ok'）
        image_label (int): 画像ラベル
        
    戻り値:
        dict: 欠陥情報辞書
    """
    base_name = image_name.replace(".jpg", "")
    defect_filename = f"defect_{defect_number}.png"
    
    info = {
        'image_name': image_name,
        'defect_image_name': defect_filename,
        'defect_image_edge_path': os.path.join(image_type, base_name, "edge", defect_filename),
        'defect_image_original_path': os.path.join(image_type, base_name, "original", defect_filename),
        'Image_label': image_label,
        'defect_label': 0
    }
    info.update(defect)
    return info

def save_defects(filtered_results, base_output_dir, image_label):
    """
    欠陥検出結果を保存します
    
    引数:
        filtered_results (list): フィルタリング結果のリスト
        base_output_dir (str): 基本出力ディレクトリ
        image_label (int): 画像ラベル
        
    戻り値:
        list: 全欠陥情報のリスト
    """
    all_defects_info = []
    image_type = "ng" if image_label == 1 else "ok"
    
    for shape_image, combined_image, large_defects, small_defects, filename in filtered_results:
        # 出力ディレクトリの作成
        base_dir = os.path.join(base_output_dir, image_type, filename.replace(".jpg", ""))
        edge_dir = os.path.join(base_dir, "edge")
        original_dir = os.path.join(base_dir, "original")
        os.makedirs(edge_dir, exist_ok=True)
        os.makedirs(original_dir, exist_ok=True)
        
        # 全ての欠陥を処理
        defect_number = 1
        for defects in [large_defects, small_defects]:
            for defect in defects:
                # エッジ画像の保存
                edge_filename = save_defect_roi(combined_image, defect, edge_dir, defect_number)
                
                # オリジナル画像の保存
                original_filename = save_defect_roi(shape_image, defect, original_dir, defect_number)
                
                # 欠陥情報の作成
                defect_info = create_defect_info(defect, filename, defect_number, 
                                               image_type, image_label)
                all_defects_info.append(defect_info)
                
                defect_number += 1
    
    return all_defects_info

def save_to_csv(defects_data, output_dir):
    """
    欠陥情報をCSVファイルに保存します
    
    引数:
        defects_data (list): 欠陥情報のリスト
        output_dir (str): 出力ディレクトリ
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "defects_data.csv")
    
    # DataFrameの作成と保存
    df = pd.DataFrame(defects_data)
    
    if os.path.exists(csv_path):
        # 既存ファイルへの追記
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"Appended defects data to: {csv_path}")
    else:
        # 新規ファイルの作成
        df.to_csv(csv_path, index=False)
        print(f"Created new defects data file: {csv_path}")

# メインの処理実行
output_dir = os.path.join(output_data_dir, "defect_images")
os.makedirs(output_dir, exist_ok=True)

# 欠陥候補の保存
all_defects_data = []
all_defects_data.extend(save_defects(filtered_ng_results, output_dir, 1))
#all_defects_data.extend(save_defects(filtered_ok_results, output_dir, 0))

# CSVファイルへの保存
csv_output_dir = os.path.join(output_data_dir, "defect_data")
save_to_csv(all_defects_data, csv_output_dir)
```

このコードでは：
1. 各機能を独立した関数に分割
2. 明確な責任分担
3. 重複コードの削減
4. 詳細なドキュメント化
5. エラー処理の改善

を行っています。

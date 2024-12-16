def process_images(processed_images, shifts):
    """
    全画像に対して欠陥検出を実行します
    
    引数:
        processed_images (list): 処理済み画像のリスト
        shifts (list): シフト量のリスト
        
    戻り値:
        list: (画像, 検出結果, Canny結果, DoG結果, ファイル名)のリスト
    """
    defect_results = []
    for (shape_path, mask, filename) in processed_images:
        # Shape画像の読み込み
        shape_path = shape_path  # パスはすでにShape画像のパス
        shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        
        if shape_image is not None:
            # 欠陥検出を実行（Shape画像に対して）
            combined, large, small = detect_defects(shape_image, mask)
            defect_results.append((shape_image, combined, large, small, filename))
        else:
            print(f"Shape画像の読み込みに失敗: {shape_path}")
            
    return defect_results

# 欠陥検出の実行
defect_ng_images = process_images(processed_ng_images, shifts)
#defect_ok_images = process_images(processed_ok_images, shifts)

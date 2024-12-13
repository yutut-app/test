# 6. エッジ補完

本セクションでは、検出された欠陥候補のエッジを補完し、より正確な欠陥領域を特定する処理について説明する。エッジ補完は以下の3つの主要な処理で構成される：

1. マスクエッジの処理
2. エッジの細線化と接続
3. エッジの統合

## 関数の詳細説明

### create_mask_edge_margin()
マスクのエッジ部分に余裕幅を持たせる関数である：

1. エッジ検出
   - Cannyエッジ検出を使用
   - パラメータ：
     - mask_edge_min_threshold：エッジ検出の最小閾値
     - mask_edge_max_threshold：エッジ検出の最大閾値

2. 余裕幅の生成
   - 検出されたエッジを膨張処理で拡張
   - パラメータ：
     - mask_edge_margin：余裕幅のサイズ（ピクセル単位）

### complete_edges()
エッジの途切れを補完し、連続的なエッジを生成する関数である：

1. 前処理
   - マスクエッジの余裕幅を生成
   - スケルトン化によるエッジの細線化

2. ノイズ除去
   - モルフォロジー演算（オープニング）の適用
   - パラメータ：
     - edge_kernel_size：カーネルサイズ
     - edge_open_iterations：オープニング処理の繰り返し回数

3. エッジの接続
   - モルフォロジー演算（クロージング）の適用
   - パラメータ：
     - edge_close_iterations：クロージング処理の繰り返し回数

4. 結果の統合
   - 元のエッジと補完したエッジの統合
   - マスクエッジ部分は元のエッジを優先

### process_defect_edges()
全ての欠陥検出結果に対してエッジ補完を実行する関数である：

1. 処理対象
   - 統合結果のエッジ
   - Cannyによる大きな欠陥のエッジ
   - DoGによる小さな欠陥のエッジ

2. 処理手順
   - 各エッジに対して個別に補完処理を実行
   - 処理結果をリストとして保持

## 可視化関数

### visualize_completed_edges()
エッジ補完の結果を可視化する関数である：

1. 表示内容
   - 元の画像
   - 補完済みのCannyエッジ結果
   - 補完済みのDoG結果
   - 補完済みの統合結果

2. 表示形式
   - 2×2のサブプロット構成
   - グレースケールでの表示
   - ファイル名をタイトルとして表示

## パラメータ調整のポイント

1. マスクエッジの処理
   - mask_edge_min_threshold, mask_edge_max_thresholdでエッジ検出感度を調整
   - mask_edge_marginで余裕幅の大きさを調整

2. エッジの補完
   - edge_kernel_sizeでモルフォロジー演算の範囲を調整
   - edge_open_iterationsでノイズ除去の強度を調整
   - edge_close_iterationsでエッジ接続の強度を調整

これらのパラメータは、ワークの形状や欠陥の特徴に応じて適切に調整する必要がある。
    sigma_pairs = [(1.5, 3.5), (2.0, 4.0), (1.0, 2.5)]
    
    for sigma1, sigma2 in sigma_pairs:
        dog_result = apply_dog_filter(image, dog_ksize, sigma1, sigma2)
        normalized = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dog_results.append(normalized)
    
    combined_dog = np.maximum.reduce(dog_results)
    
    # 各種マスクの生成
    binary_dog = apply_dynamic_threshold(image)
    contrast_mask = calculate_contrast_mask(image)
    gradient_magnitude = calculate_gradient_magnitude(image)
    
    # マスクの統合
    bright_region = cv2.bitwise_and(bright_mask, gradient_magnitude)
    dark_region = cv2.bitwise_and(dark_mask, gradient_magnitude)
    combined_mask = cv2.bitwise_or(bright_region, dark_region)
    
    # 結果の統合
    result = cv2.bitwise_and(combined_mask, combined_dog)
    result = cv2.bitwise_and(result, contrast_mask)
    result = cv2.bitwise_and(result, binary_dog)
    result = cv2.bitwise_and(result, mask)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                     (dog_merge_distance, dog_merge_distance))
    merged_result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    # 明るい領域の追加
    final_result = cv2.bitwise_or(merged_result, cv2.bitwise_and(bright_mask, mask))
    
    return final_result

def calculate_gradient_magnitude(image):
    """
    画像の勾配強度を計算します
    
    引数:
        image (numpy.ndarray): 入力画像
        
    戻り値:
        numpy.ndarray: 勾配強度画像
    """
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def detect_defects(image, mask):
    """
    CannyとDoGを組み合わせて欠陥検出を行います
    
    引数:
        image (numpy.ndarray): 入力画像
        mask (numpy.ndarray): マスク画像
        
    戻り値:
        tuple: (統合結果, Canny結果, DoG結果)
    """
    large_defects = apply_canny_detection(image, mask)
    small_defects = apply_dog_detection(image, mask)
    combined_result = cv2.bitwise_or(large_defects, small_defects)
    
    return combined_result, large_defects, small_defects

def process_images(processed_images):
    """
    全画像に対して欠陥検出を実行します
    
    引数:
        processed_images (list): 処理済み画像のリスト
        
    戻り値:
        list: (画像, 検出結果, Canny結果, DoG結果, ファイル名)のリスト
    """
    defect_results = []
    for shape_image, mask, filename in processed_images:
        combined, large, small = detect_defects(shape_image, mask)
        defect_results.append((shape_image, combined, large, small, filename))
    return defect_results

# 欠陥検出の実行
defect_ng_images = process_images(processed_ng_images)
#defect_ok_images = process_images(processed_ok_images)

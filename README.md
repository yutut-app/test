まず最初の部分を送ります。

```python
# 5. Canny+DoGによる欠陥候補検出

def detect_bright_dark_regions(image):
    """
    画像から明るい領域と暗い領域を検出します
    
    引数:
        image (ndarray): 入力画像
    
    戻り値:
        tuple: 明るい領域のマスクと暗い領域のマスク
    """
    _, bright_mask = cv2.threshold(image, bright_threshold, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(image, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    return bright_mask, dark_mask

def compute_gradient_magnitude(image):
    """
    画像の勾配強度を計算します
    
    引数:
        image (ndarray): 入力画像
    
    戻り値:
        ndarray: 正規化された勾配強度マップ
    """
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def detect_edges_and_texture(shape_image, mask):
    """
    Cannyエッジ検出とテクスチャ検出により大きな鋳巣を検出します
    
    引数:
        shape_image (ndarray): Shape画像
        mask (ndarray): マスク画像
    
    戻り値:
        ndarray: 検出結果のマスク
    """
    # マスク領域内の画像を取得
    masked_image = cv2.bitwise_and(shape_image, shape_image, mask=mask)
    
    # 明るい部分と暗い部分の検出
    bright_mask, dark_mask = detect_bright_dark_regions(shape_image)
    
    # Cannyエッジ検出
    blurred_image = cv2.GaussianBlur(masked_image, canny_kernel_size, canny_sigma)
    edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
    
    # テクスチャ検出
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    laplacian_edges = np.uint8(np.absolute(laplacian) > texture_threshold) * 255
    
    # 勾配強度の計算
    gradient_magnitude = compute_gradient_magnitude(masked_image)
    
    # 各検出結果の統合
    combined_bright = cv2.bitwise_and(bright_mask, gradient_magnitude)
    combined_dark = cv2.bitwise_and(dark_mask, gradient_magnitude)
    
    combined_edges = cv2.bitwise_or(edges, laplacian_edges)
    combined_edges = cv2.bitwise_or(combined_edges, combined_bright)
    combined_edges = cv2.bitwise_or(combined_edges, combined_dark)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (canny_merge_distance, canny_merge_distance))
    merged_result = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
    
    # マスク適用と最終マスク作成
    masked_result = cv2.bitwise_and(merged_result, merged_result, mask=mask)
    final_mask = cv2.bitwise_or(masked_result, cv2.bitwise_and(bright_mask, mask))
    
    return final_mask
```

続きは次のメッセージで送ります。主な改良点：
1. 共通処理を独立した関数に分割
2. 各関数に詳細なdocstrings追加
3. 処理ステップごとに適切なコメント追加
4. 変数名をより明確に変更
5. 処理の流れを整理

次のメッセージでは残りの関数とビジュアライゼーション部分を送ります。

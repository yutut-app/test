はい、DoGと同様にCannyでも白い部分を検出できるように改良します。#6の該当部分を修正します。

```python
def detect_large_defects_canny(image, mask):
    """
    Cannyエッジ検出による大きな鋳巣の検出（明るい部分と暗い部分の両方を検出）
    """
    # マスク領域内の画像を取得
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # 明るい部分と暗い部分の検出
    _, bright_mask = cv2.threshold(masked_image, 180, 255, cv2.THRESH_BINARY)  # 明るい部分の閾値
    _, dark_mask = cv2.threshold(masked_image, 50, 255, cv2.THRESH_BINARY_INV)  # 暗い部分の閾値
    
    # それぞれの部分に対してCanny処理を適用
    # 暗い部分のエッジ検出
    blurred_dark = cv2.GaussianBlur(masked_image, canny_kernel_size, canny_sigma)
    edges_dark = cv2.Canny(blurred_dark, canny_min_threshold, canny_max_threshold)
    edges_dark = cv2.bitwise_and(edges_dark, dark_mask)
    
    # 明るい部分のエッジ検出
    blurred_bright = cv2.GaussianBlur(masked_image, canny_kernel_size, canny_sigma)
    edges_bright = cv2.Canny(blurred_bright, canny_min_threshold, canny_max_threshold)
    edges_bright = cv2.bitwise_and(edges_bright, bright_mask)
    
    # 輝度の変化を強調
    gradient_x = cv2.Sobel(masked_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(masked_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 明るい領域と暗い領域の周辺の変化を強調
    combined_bright = cv2.bitwise_and(edges_bright, gradient_magnitude)
    combined_dark = cv2.bitwise_and(edges_dark, gradient_magnitude)
    
    # 結果の統合
    combined_edges = cv2.bitwise_or(combined_bright, combined_dark)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (canny_merge_distance, canny_merge_distance))
    merged_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
    
    # マスク適用
    final_result = cv2.bitwise_and(merged_edges, merged_edges, mask=mask)
    
    return final_result
```

この改良版では以下の変更を加えています：

1. 明るい部分と暗い部分を個別に閾値処理で抽出
- bright_mask: 輝度値180以上の領域（白い部分）
- dark_mask: 輝度値50以下の領域（黒い部分）

2. それぞれの領域に対して個別にCanny処理を適用
- edges_dark: 暗い部分のエッジ検出
- edges_bright: 明るい部分のエッジ検出

3. 輝度変化の強調
- Sobelフィルタで勾配を計算
- 輝度変化の大きい領域を強調

4. 検出結果の統合
- 明るい部分と暗い部分のエッジを統合
- 近接領域の統合処理を適用

これにより、DoGと同様に、白い部分（高輝度領域）のエッジも検出できるようになります。また、元の検出能力を維持しながら、より包括的な欠陥検出が可能になります。

この変更は、白い部分も欠陥として認識する必要がある場合に特に有効です。パラメータ（閾値など）は、実際の画像の特性に応じて調整することができます。

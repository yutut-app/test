すみません。`bbox`プロパティの使用方法を間違えていました。`measure.regionprops`で得られる領域の最小・最大座標を正しく取得するように修正します。

```python
def save_defect_image(edge_image, original_image, defect, output_dir_edge, output_dir_original, image_name, defect_number):
    """
    欠陥候補領域の画像を保存（エッジ検出画像とオリジナル画像の両方）
    """
    # defectから直接x, y, width, heightを取得
    x = defect['x']
    y = defect['y']
    width = defect['width']
    height = defect['height']
    
    # 余白を1ピクセル追加（ただし画像の範囲内に収める）
    x1_with_margin = max(x - 1, 0)
    y1_with_margin = max(y - 1, 0)
    x2_with_margin = min(x + width + 1, edge_image.shape[1])
    y2_with_margin = min(y + height + 1, edge_image.shape[0])
    
    # エッジ検出画像の切り出しと保存
    defect_edge = edge_image[y1_with_margin:y2_with_margin, x1_with_margin:x2_with_margin]
    enlarged_edge = cv2.resize(defect_edge, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    edge_filename = f"defect_{defect_number}.png"
    edge_path = os.path.join(output_dir_edge, edge_filename)
    cv2.imwrite(edge_path, enlarged_edge)
    
    # オリジナル画像の切り出しと保存
    defect_original = original_image[y1_with_margin:y2_with_margin, x1_with_margin:x2_with_margin]
    enlarged_original = cv2.resize(defect_original, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    original_filename = f"defect_{defect_number}.png"
    original_path = os.path.join(output_dir_original, original_filename)
    cv2.imwrite(original_path, enlarged_original)
    
    return edge_filename, original_filename
```

変更点：
- `bbox`プロパティの代わりに、すでにdefectに格納されている'x', 'y', 'width', 'height'を使用
- これらの値から切り出し範囲を計算し、1ピクセルの余白を追加

これで正しく動作するはずです。

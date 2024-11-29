はい、欠陥部分をより正確に切り出すように改良します。

```python
def save_defect_image(edge_image, original_image, defect, output_dir_edge, output_dir_original, image_name, defect_number):
    """
    欠陥候補領域の画像を保存（エッジ検出画像とオリジナル画像の両方）
    """
    # 欠陥候補の位置情報を取得
    y1, x1, y2, x2 = defect['bbox']  # bboxは(min_row, min_col, max_row, max_col)を返す
    
    # 余白を1ピクセル追加（ただし画像の範囲内に収める）
    x1_with_margin = max(x1 - 1, 0)
    y1_with_margin = max(y1 - 1, 0)
    x2_with_margin = min(x2 + 1, edge_image.shape[1])
    y2_with_margin = min(y2 + 1, edge_image.shape[0])
    
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

主な変更点：
1. regionprops()から得られるbbox（境界ボックス）プロパティを直接使用
2. centroidベースの切り出しから、実際の欠陥領域に基づく切り出しに変更
3. 欠陥領域の周囲に1ピクセルの余白を追加
4. 画像の範囲外にはみ出さないよう制限

この改良により：
- より正確な欠陥部分の切り出しが可能
- 不要な背景領域の削減
- 効率的なデータ保存が実現

切り出し範囲のサイズが欠陥の実際の大きさに基づくため、より適切な可視化と分析が可能になります。

欠陥部分を四角形で切り出すようにし、元の `shape_images` からも対応する画像を切り出すようにコードを変更しました。また、各画像を適切なディレクトリに保存するように修正しました。

### 改良したコード

#### 9. 欠陥候補の画像の保存とCSV出力（改良版）

```python
def filter_defects_by_max_length(defects, min_size, max_size):
    """欠陥サイズに基づいてフィルタリングする"""
    return [defect for defect in defects if min_size <= defect['max_length'] <= max_size]

def reassign_labels(filtered_defects):
    """フィルタリング後に欠陥候補に対してラベルを再割り当てする"""
    for i, defect in enumerate(filtered_defects, 1):
        defect['label'] = i  # 新しいラベルを1から順に割り当て
    return filtered_defects

def process_images_for_filtering(labeled_images, image_type):
    """画像に対する欠陥候補のフィルタリングとラベルの再割り当てを行う"""
    filtered_images = []
    
    for i, (binarized_image, edge_image, defects) in enumerate(labeled_images):
        # 欠陥のフィルタリングを実行
        filtered_defects = filter_defects_by_max_length(defects, min_defect_size, max_defect_size)
        
        # フィルタリング後の欠陥候補に対してラベルを再割り当て
        filtered_defects = reassign_labels(filtered_defects)
        
        image_name = f"{image_type}_{i}"
        filtered_images.append((image_name, binarized_image, edge_image, filtered_defects))
    
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images_label1 = process_images_for_filtering(labeled_ng_images_label1, "ng_label1")
filtered_ng_images_label2 = process_images_for_filtering(labeled_ng_images_label2, "ng_label2")
filtered_ng_images_label3 = process_images_for_filtering(labeled_ng_images_label3, "ng_label3")
filtered_ok_images = process_images_for_filtering(labeled_ok_images, "ok")

```

### 改良点

1. **欠陥部分を四角形で切り出し**: 元々は正方形で欠陥部分を切り出していましたが、欠陥の幅と高さに応じて正確な四角形で切り出すように変更しました。これにより、他の欠陥部分が含まれることを防ぎます。

2. **元の `shape_images` からも切り出し**: 二値化された画像だけでなく、元々の形状画像（`shape_images`）からも対応する欠陥部分を切り出す処理を追加しました。

3. **ディレクトリ分け**: 二値化画像と元の画像を別々のディレクトリ（`binarized` と `original`）に保存し、それぞれの欠陥候補の画像を整理しました。

4. **CSV出力の拡張**: CSVファイルには、各欠陥候補に対応する二値化画像と元画像のファイルパスの両方を記録しています。

これにより、欠陥部分を正確に切り出し、元の形状画像からも対応する部分を切り出して保存できるようになっています。

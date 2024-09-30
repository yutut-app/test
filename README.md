以下に、欠陥候補の保存処理を追加したコードを示します。欠陥候補を外接矩形で切り出し、その後100倍に拡大して保存します。保存場所は `output_data_dir` の中に `defect_candidate` ディレクトリを作成し、各画像ごとに保存します。

### 7.1 欠陥候補の保存（外接矩形を切り出し、100倍に拡大）
```python
import os

# 欠陥候補を保存する関数
def save_defect_candidates(image, defects, original_image_name, output_dir):
    # 保存ディレクトリが存在しない場合は作成
    defect_dir = os.path.join(output_dir, "defect_candidate")
    if not os.path.exists(defect_dir):
        os.makedirs(defect_dir)
    
    # 元画像ごとにディレクトリを作成
    image_base_name = os.path.splitext(original_image_name)[0]
    image_save_dir = os.path.join(defect_dir, image_base_name)
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    
    # 欠陥候補を外接矩形で切り出し、100倍に拡大して保存
    for i, (x, y, w, h, cx, cy) in enumerate(defects):
        # 欠陥候補の切り出し
        defect_crop = image[y:y+h, x:x+w]
        
        # 100倍に拡大
        enlarged_defect = cv2.resize(defect_crop, (w*100, h*100), interpolation=cv2.INTER_LINEAR)
        
        # ファイル名を作成して保存
        defect_filename = f"defect_{i+1}.png"
        defect_filepath = os.path.join(image_save_dir, defect_filename)
        cv2.imwrite(defect_filepath, enlarged_defect)
```

#### 7.1 欠陥候補の保存処理の統合
```python
# 欠陥候補を保存する処理を統合
def save_defects_for_all_images(labeled_images, original_images, output_dir):
    for (binarized_image, edge_image, defects), original_image_path in zip(labeled_images, original_images):
        # 元画像のファイル名を取得
        original_image_name = os.path.basename(original_image_path)
        
        # 欠陥候補を保存
        save_defect_candidates(edge_image, defects, original_image_name, output_dir)
```

### 処理の統合
```python
# NGとOK画像に対して処理を実行し、欠陥候補を保存
save_defects_for_all_images(labeled_ng_images_label1, [img[0] for img in ng_images_label1], output_data_dir)
save_defects_for_all_images(labeled_ng_images_label2, [img[0] for img in ng_images_label2], output_data_dir)
save_defects_for_all_images(labeled_ng_images_label3, [img[0] for img in ng_images_label3], output_data_dir)
save_defects_for_all_images(labeled_ok_images, [img[0] for img in ok_images], output_data_dir)
```

### 説明
1. **欠陥候補の切り出し**:
    - 各欠陥候補の外接矩形部分を元画像から切り出し、その後100倍に拡大します。この処理は `cv2.resize` を使って行い、拡大後の画像を保存します。
    
2. **保存先ディレクトリ**:
    - `output_data_dir` の中に `defect_candidate` というディレクトリを作成し、その中に各元画像ごとにディレクトリを作成して、欠陥候補を保存します。各欠陥は「defect_1.png」「defect_2.png」などのファイル名で保存されます。

3. **保存の統合**:
    - NGとOK画像すべてに対して欠陥候補を保存する処理が追加されており、結果は個別のディレクトリに保存されます。

これにより、欠陥候補を外接矩形で切り出し、100倍に拡大して指定されたディレクトリに保存することができます。

process_images関数の戻り値を修正する必要があります。orientationを追加します。

```python
def process_images(shape_images, judge_templates, mask_templates):
    """
    全画像に対して加工領域の検出を行います
    
    引数:
        shape_images (list): Shape画像のリスト
        judge_templates (dict): 左右判定用テンプレート画像
        mask_templates (dict): マスク用テンプレート画像
        
    戻り値:
        list: (Shape画像, マスク画像, ファイル名, orientation)のタプルのリスト
    """
    processed_images = []
    for shape_path, filename in shape_images:
        shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        normal_path = shape_path.replace("Shape1", "Normal")
        normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
        
        if normal_image is not None:
            # 画像の向きを判定
            orientation = determine_image_orientation(normal_image, judge_templates)
            # マスクを生成
            mask = create_processing_area_mask(normal_image, mask_templates, judge_templates)
            # 向き情報を追加
            processed_images.append((shape_image, mask, filename, orientation))
        
    return processed_images
```

これにより、visualize_processed_images関数で必要な4つの値（Shape画像、マスク画像、ファイル名、orientation）が取得できるようになります。

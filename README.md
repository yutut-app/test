はい、問題を修正しましょう。#4と#5の処理の連携を修正して、Shape1画像のみを使用するように変更します。

まず、#4の`process_images`関数の修正版を示します：

```python
def process_images(shape_images, judge_templates, mask_templates):
   """
   全画像に対して加工領域の検出を行います
   
   引数:
       shape_images (list): Shape画像のリスト
       judge_templates (dict): 左右判定用テンプレート画像
       mask_templates (dict): マスク用テンプレート画像
       
   戻り値:
       tuple: (処理済み画像のリスト, シフト量のリスト)
   """
   processed_images = []
   shifts = []
   
   for shape_path, filename in shape_images:
       # Normal画像のパスを生成
       normal_path = shape_path.replace("Shape1", "Normal")
       
       # 両方の画像を読み込み
       shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
       normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
       
       if shape_image is not None and normal_image is not None:
           # Normalイメージで向きを判定し、マスクを作成
           # 二値化
           _, binary_image = cv2.threshold(normal_image, binary_threshold, 255, cv2.THRESH_BINARY)
           
           # 向きを判定
           orientation = determine_image_orientation(normal_image, judge_templates)
           template = mask_templates[orientation]
           
           # 位相限定相関によるズレ計算
           img_float = np.float32(binary_image)
           template_float = np.float32(template)
           shift, _ = cv2.phaseCorrelate(img_float, template_float)
           dx, dy = shift
           
           # マスク生成
           rows, cols = template.shape
           M = np.float32([[1, 0, dx], [0, 1, dy]])
           mask = cv2.warpAffine(template, M, (cols, rows))
           
           # Shape画像とマスクを保存
           processed_images.append((shape_image, mask, filename))
           shifts.append((dx, dy))
       else:
           print(f"画像の読み込みに失敗: {filename}")
   
   return processed_images, shifts
```

次に、#5の`process_images`関数を修正します：

```python
def process_images(processed_images, shifts):
    """
    全画像に対して欠陥検出を実行します
    
    引数:
        processed_images (list): 処理済み画像のリスト（Shape画像とマスク）
        shifts (list): シフト量のリスト
        
    戻り値:
        list: (画像, 検出結果, Canny結果, DoG結果, ファイル名)のリスト
    """
    defect_results = []
    for (shape_image, mask, filename) in processed_images:
        # 直接Shape画像に対して欠陥検出を実行
        combined, large, small = detect_defects(shape_image, mask)
        defect_results.append((shape_image, combined, large, small, filename))
            
    return defect_results
```

主な修正点：
1. #4の`process_images`で、Shape1画像を直接メモリに読み込んで保持
2. #5の`process_images`では、既に読み込まれているShape1画像を使用
3. パスの変換や追加の画像読み込みを排除

これにより：
- Normal画像はマスク生成にのみ使用
- 欠陥検出はShape1画像のみで実行
- 可視化時も正しい画像（Shape1）が表示される

という望ましい動作が実現されます。

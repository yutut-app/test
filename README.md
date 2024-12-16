# 4. 加工領域の特定
まず、パラメータ設定の修正から行います。

```python
# 2. パラメータ設定（修正版）

# ディレクトリとファイルパス
input_data_dir = r"../../data/input"
output_data_dir = r"../../data/output"
left_right_judge_template_dir = os.path.join(input_data_dir, "left_right_judge_template")
mask_template_dir = os.path.join(input_data_dir, "mask_template")

# 撮影画像の左右判定用テンプレートファイルパス
right_judge_template_path = os.path.join(left_right_judge_template_dir, "right_template.jpg")
left_judge_template_path = os.path.join(left_right_judge_template_dir, "left_template.jpg")

# マスク用テンプレートファイルパス
right_mask_template_path = os.path.join(mask_template_dir, "right_template.jpg")
left_mask_template_path = os.path.join(mask_template_dir, "left_template.jpg")

# テンプレートマッチングパラメータ
template_match_threshold = 0.8  # マッチング判定の閾値（0-1、大きいほど厳密）

# 二値化パラメータ
binary_threshold = 128  # 二値化の閾値
max_binary_value = 255  # 二値化後の最大値

# マスクパラメータ
mask_threshold = 128  # マスク生成時の閾値
```

続いて、加工領域特定の実装を行います：

```python
def determine_image_orientation(image, judge_templates):
    """
    画像が左右どちらの撮影画像かを判定します
    
    引数:
        image (numpy.ndarray): 入力画像
        judge_templates (dict): 左右判定用テンプレート画像
        
    戻り値:
        str: 'left' または 'right'
    """
    # テンプレートマッチング実行
    left_result = cv2.matchTemplate(image, judge_templates['left'], cv2.TM_CCOEFF_NORMED)
    right_result = cv2.matchTemplate(image, judge_templates['right'], cv2.TM_CCOEFF_NORMED)
    
    # 最大値を比較して判定
    left_val = np.max(left_result)
    right_val = np.max(right_result)
    
    return 'right' if right_val > left_val else 'left'

def align_images(image, template):
    """
    位相限定相関法を用いて画像の位置合わせを行います
    
    引数:
        image (numpy.ndarray): 入力画像
        template (numpy.ndarray): テンプレート画像
        
    戻り値:
        numpy.ndarray: 位置合わせ後の画像
    """
    # float32型に変換
    img_float = np.float32(image)
    template_float = np.float32(template)
    
    # 位相限定相関の計算
    shift, _ = cv2.phaseCorrelate(img_float, template_float)
    dx, dy = shift
    
    # 移動行列の作成と適用
    rows, cols = image.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned_image = cv2.warpAffine(image, M, (cols, rows))
    
    return aligned_image

def create_processing_area_mask(image, mask_templates, judge_templates):
    """
    加工領域のマスクを作成します
    
    引数:
        image (numpy.ndarray): 入力画像
        mask_templates (dict): マスク用テンプレート画像
        judge_templates (dict): 左右判定用テンプレート画像
        
    戻り値:
        numpy.ndarray: 加工領域のマスク画像
    """
    # 画像の向きを判定
    orientation = determine_image_orientation(image, judge_templates)
    template = mask_templates[orientation]
    
    # 入力画像の二値化
    _, binary_image = cv2.threshold(image, binary_threshold, max_binary_value, cv2.THRESH_BINARY)
    
    # 位置合わせ
    aligned_image = align_images(binary_image, template)
    
    # マスクの生成（テンプレートの白い部分を抽出）
    _, mask = cv2.threshold(template, mask_threshold, max_binary_value, cv2.THRESH_BINARY)
    
    return mask

def process_images(shape_images, judge_templates, mask_templates):
    """
    全画像に対して加工領域の検出を行います
    
    引数:
        shape_images (list): Shape画像のリスト
        judge_templates (dict): 左右判定用テンプレート画像
        mask_templates (dict): マスク用テンプレート画像
        
    戻り値:
        list: (Shape画像, マスク画像, ファイル名)のタプルのリスト
    """
    processed_images = []
    for shape_path, filename in shape_images:
        shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        mask = create_processing_area_mask(shape_image, mask_templates, judge_templates)
        processed_images.append((shape_image, mask, filename))
    
    return processed_images
```

可視化関数の実装は変更なしで使用できます。

これらの実装における主な変更点：

1. 円検出処理の削除
2. 位置合わせ手法の変更（アフィン変換から位相限定相関法へ）
3. マスク生成の簡略化
4. スケール調整処理の削除
5. テンプレート反転処理の削除

この実装により、要件に沿った加工領域の特定が可能となります。

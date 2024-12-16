# 4. 加工領域の特定
ご指示の内容に基づき、#4加工領域の特定の処理を修正します。まず、パラメータ設定の修正から行います。

```python
# 2. パラメータの設定

# ディレクトリとファイルパス
input_data_dir = r"../../data/input"
output_data_dir = r"../../data/output"
left_right_judge_template_dir = os.path.join(input_data_dir, "left_right_judge_template")
mask_template_dir = os.path.join(input_data_dir, "mask_template")

# 撮影画像がワークの左側の画像か右側の画像か判断用テンプレートファイルパス
right_judge_template_path = os.path.join(left_right_judge_template_dir, "right_template.jpg")
left_judge_template_path = os.path.join(left_right_judge_template_dir, "left_template.jpg")

# マスク用テンプレートファイルパス
right_mask_template_path = os.path.join(mask_template_dir, "right_template.jpg")
left_mask_template_path = os.path.join(mask_template_dir, "left_template.jpg")

# テンプレートマッチングパラメータ
template_match_threshold = 0.8  # マッチング判定の閾値（0-1、大きいほど厳密）

# 二値化パラメータ
binary_threshold = 128  # 二値化の閾値（0-255）
```

次に、加工領域の特定の処理を修正します：

```python
def determine_image_orientation(image, judge_templates):
    """
    画像がワークの左側の撮影画像か右側の撮影画像かをテンプレートより判定します
    
    引数:
        image (numpy.ndarray): 入力画像
        judge_templates (dict): 左右判定用テンプレート画像
        
    戻り値:
        str: 'left' または 'right'
    """
    # テンプレートマッチング実行
    left_result = cv2.matchTemplate(image, judge_templates['left'], cv2.TM_CCOEFF_NORMED)
    right_result = cv2.matchTemplate(image, judge_templates['right'], cv2.TM_CCOEFF_NORMED)
    
    # マッチング度の最大値を比較
    left_val = np.max(left_result)
    right_val = np.max(right_result)
    
    return 'right' if right_val > left_val else 'left'

def create_processing_area_mask(normal_image, mask_templates, judge_templates):
    """
    加工領域のマスクを作成します
    
    引数:
        normal_image (numpy.ndarray): Normal画像
        mask_templates (dict): マスク用テンプレート画像
        judge_templates (dict): 左右判定用テンプレート画像
        
    戻り値:
        numpy.ndarray: 加工領域のマスク画像
    """
    try:
        # 画像の二値化
        _, binary_image = cv2.threshold(normal_image, binary_threshold, 255, cv2.THRESH_BINARY)
        
        # 画像の向きを判定
        orientation = determine_image_orientation(normal_image, judge_templates)
        template = mask_templates[orientation]
        
        # float32型に変換
        img_float = np.float32(binary_image)
        template_float = np.float32(template)
        
        # 位相限定相関によるズレ計算
        shift, _ = cv2.phaseCorrelate(img_float, template_float)
        dx, dy = shift
        
        # ズレ補正行列の作成と適用
        rows, cols = template.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_template = cv2.warpAffine(template, M, (cols, rows))
        
        return aligned_template
        
    except Exception as e:
        print(f"マスク作成エラー: {e}")
        return np.ones_like(normal_image) * 255

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
        # Shape画像とNormal画像のパスを生成
        normal_path = shape_path.replace("Shape1", "Normal")
        
        # 画像読み込み
        shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
        
        if shape_image is not None and normal_image is not None:
            # マスク生成
            mask = create_processing_area_mask(normal_image, mask_templates, judge_templates)
            processed_images.append((shape_image, mask, filename))
        else:
            print(f"画像の読み込みに失敗: {filename}")
    
    return processed_images
```

主な変更点：
1. 円検出関連の処理を削除
2. アフィン変換を位相限定相関による位置合わせに変更
3. スケール調整関連のパラメータを削除
4. テンプレート反転処理を削除
5. 位置合わせ処理をシンプル化
6. Normal画像とShape1画像の対応付けを追加

可視化関数は既存のものをそのまま使用できます。この修正により、より簡潔で効率的な加工領域の特定が可能になります。

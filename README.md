```python
# 4. 加工領域の特定(マスク生成)

def template_matching(image, template_path):
    """
    テンプレートマッチングを実行します
    
    引数:
        image (ndarray): 入力画像
        template_path (str): テンプレート画像のパス
    
    戻り値:
        tuple: マッチングスコアと位置
    """
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc

def judge_work_direction(normal_image):
    """
    ワークの撮影方向（右側/左側）を判断します
    
    引数:
        normal_image (ndarray): Normal画像
    
    戻り値:
        str: 'right'または'left'
    """
    right_val, _ = template_matching(normal_image, right_judge_template_path)
    left_val, _ = template_matching(normal_image, left_judge_template_path)
    return 'right' if right_val > left_val else 'left'

def create_mask_from_template(shape_image, mask_template_path):
    """
    マスクテンプレートと位相限定相関によりマスクを作成します
    
    引数:
        shape_image (ndarray): Shape画像
        mask_template_path (str): マスクテンプレート画像のパス
    
    戻り値:
        ndarray: 生成されたマスク画像
    """
    # マスクテンプレートの読み込み
    mask_template = cv2.imread(mask_template_path, cv2.IMREAD_GRAYSCALE)
    
    # 位相限定相関のための型変換
    shape_float = np.float32(shape_image)
    template_float = np.float32(mask_template)
    
    # ズレ量の計算
    shift, _ = cv2.phaseCorrelate(shape_float, template_float)
    dx, dy = shift
    
    # マスクテンプレートのズレ補正
    rows, cols = shape_image.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    corrected_mask = cv2.warpAffine(mask_template, M, (cols, rows))
    
    # 二値化とモルフォロジー処理
    _, binary_mask = cv2.threshold(corrected_mask, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
    
    return binary_mask, mask_template, corrected_mask

def process_images_with_mask(image_pairs):
    """
    全画像に対してマスク生成処理を実行します
    
    引数:
        image_pairs (list): 画像ペアのリスト
    
    戻り値:
        list: 処理結果のリスト（マスク画像、Shape画像、ファイル名、マスクテンプレート、補正マスク）
    """
    processed_images = []
    for normal_path, shape_path, original_filename in image_pairs:
        # 画像の読み込み
        normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
        shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        
        # ワークの向きを判断
        direction = judge_work_direction(normal_image)
        
        # 向きに応じたマスクテンプレートを選択
        mask_template_path = right_mask_template_path if direction == 'right' else left_mask_template_path
        
        # マスク作成
        binary_mask, mask_template, corrected_mask = create_mask_from_template(shape_image, mask_template_path)
        
        processed_images.append((binary_mask, shape_image, original_filename, mask_template, corrected_mask))
    return processed_images

# NGとOK画像に対してマスク生成を実行
processed_ng_images_label1 = process_images_with_mask(ng_images_label1)
processed_ok_images = process_images_with_mask(ok_images)

print(f"処理したNG画像数: {len(processed_ng_images_label1)}")
print(f"処理したOK画像数: {len(processed_ok_images)}")
```

```python
# マスク生成結果の可視化

def visualize_mask_generation(processed_images, pair_index):
    """
    マスク生成結果を可視化します
    
    引数:
        processed_images (list): 処理済み画像のリスト
        pair_index (int): 表示するペアのインデックス
    """
    if not processed_images or pair_index >= len(processed_images):
        print("指定されたインデックスの画像が存在しません")
        return
    
    binary_mask, shape_image, filename, mask_template, corrected_mask = processed_images[pair_index]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    plt.suptitle(f'Mask Generation Result - {filename}', fontsize=16)
    
    # Shape画像
    axes[0, 0].imshow(shape_image, cmap='gray')
    axes[0, 0].set_title('Shape Image')
    axes[0, 0].axis('off')
    
    # マスクテンプレート
    axes[0, 1].imshow(mask_template, cmap='gray')
    axes[0, 1].set_title('Mask Template')
    axes[0, 1].axis('off')
    
    # 補正後のマスク
    axes[1, 0].imshow(corrected_mask, cmap='gray')
    axes[1, 0].set_title('Corrected Mask')
    axes[1, 0].axis('off')
    
    # 最終マスク
    axes[1, 1].imshow(binary_mask, cmap='gray')
    axes[1, 1].set_title('Final Mask')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# NG画像の最初のペアを表示
print("NG画像のマスク生成結果:")
if processed_ng_images_label1:
    visualize_mask_generation(processed_ng_images_label1, 0)

# OK画像の最初のペアを表示
print("\nOK画像のマスク生成結果:")
if processed_ok_images:
    visualize_mask_generation(processed_ok_images, 0)
```

主な改良点：
1. 関数の分割と明確な責任の割り当て
2. 詳細なdocstringsの追加
3. 中間結果（マスクテンプレート、補正マスク）の保持
4. 可視化機能の追加
5. エラー処理の追加
6. 変数名の明確化
7. 処理と可視化の分離

処理の流れ：
1. Normal画像で左右判定
2. Shape画像に対してマスク生成
3. 位相限定相関でズレ補正
4. 二値化とモルフォロジー処理
5. 結果の可視化（2×2のグリッド）

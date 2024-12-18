```python
# 4. 加工領域の特定（マスク生成）

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

def determine_work_side(keyence_image):
    """
    ワークが右側か左側かを判断します
    
    引数:
        keyence_image (ndarray): キーエンス画像
    
    戻り値:
        str: "right" または "left"
    """
    right_val, _ = template_matching(keyence_image, right_judge_template_path)
    left_val, _ = template_matching(keyence_image, left_judge_template_path)
    return "right" if right_val > left_val else "left"

def get_template_region(keyence_image, template_path):
    """
    テンプレートマッチングで加工部分の位置を特定します
    
    引数:
        keyence_image (ndarray): キーエンス画像
        template_path (str): テンプレート画像のパス
    
    戻り値:
        tuple: (x1, y1, x2, y2) 形式の領域座標
    """
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    h, w = template.shape
    
    res = cv2.matchTemplate(keyence_image, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    
    x, y = max_loc
    return (x, y, x + w, y + h)

def adjust_template_mask(normal_image, template_path):
    """
    位相限定相関によるテンプレートマスクの位置補正を行います
    
    引数:
        normal_image (ndarray): Normal画像
        template_path (str): マスクテンプレート画像のパス
    
    戻り値:
        ndarray: 補正済みマスク画像
    """
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    # 加工部分の位置を特定
    x1, y1, x2, y2 = get_template_region(normal_image, template_path)
    
    # 加工部分のみを切り出し
    work_region = normal_image[y1:y2, x1:x2]
    template_region = cv2.resize(template, (x2-x1, y2-y1))
    
    # 位相限定相関の計算
    work_float = np.float32(work_region)
    template_float = np.float32(template_region)
    shift, _ = cv2.phaseCorrelate(work_float, template_float)
    dx, dy = shift
    
    # ズレ補正
    rows, cols = normal_image.shape
    M = np.float32([[1, 0, dx+x1], [0, 1, dy+y1]])
    adjusted_template = cv2.warpAffine(template, M, (cols, rows))
    
    # 二値化とモルフォロジー処理
    _, adjusted_mask = cv2.threshold(adjusted_template, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    adjusted_mask = cv2.morphologyEx(adjusted_mask, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
    adjusted_mask = cv2.morphologyEx(adjusted_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
    
    return adjusted_mask, template, adjusted_template

def create_masks(image_pairs):
    """
    画像群に対してマスク生成を実行します
    
    引数:
        image_pairs (list): (Normal画像パス, Shape画像パス, ファイル名)のリスト
    
    戻り値:
        list: (マスク, Shape画像, ファイル名, テンプレート, 調整前テンプレート)のリスト
    """
    processed_images = []
    for normal_path, shape_path, original_filename in image_pairs:
        # 画像読み込み
        normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
        shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        
        # ワークの向きを判定
        work_side = determine_work_side(normal_image)
        template_path = right_mask_template_path if work_side == "right" else left_mask_template_path
        
        # マスク生成
        mask, template, adjusted_template = adjust_template_mask(normal_image, template_path)
        
        processed_images.append((mask, shape_image, original_filename, template, adjusted_template))
    
    return processed_images

# NGとOK画像に対してマスク生成を実行
masked_ng_images = create_masks(ng_images_label1)
masked_ok_images = create_masks(ok_images)
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
        
    mask, shape_image, filename, template, adjusted_template = processed_images[pair_index]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Mask Generation Results - {filename}', fontsize=16)
    
    # Shape画像
    axes[0, 0].imshow(shape_image, cmap='gray')
    axes[0, 0].set_title('Shape Image')
    axes[0, 0].axis('off')
    
    # テンプレート
    axes[0, 1].imshow(template, cmap='gray')
    axes[0, 1].set_title('Template')
    axes[0, 1].axis('off')
    
    # 移動したテンプレート
    axes[1, 0].imshow(adjusted_template, cmap='gray')
    axes[1, 0].set_title('Adjusted Template')
    axes[1, 0].axis('off')
    
    # 最終マスク
    axes[1, 1].imshow(mask, cmap='gray')
    axes[1, 1].set_title('Final Mask')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# NG画像の最初のペアを表示
print("NG画像のマスク生成結果:")
if masked_ng_images:
    visualize_mask_generation(masked_ng_images, 0)

# OK画像の最初のペアを表示
print("\nOK画像のマスク生成結果:")
if masked_ok_images:
    visualize_mask_generation(masked_ok_images, 0)
```

主な改良点：
1. 各関数にdocstringsを追加
2. 処理と可視化を分離
3. マスク生成過程の中間結果を保持
4. 変数名をより明確に
5. エラーハンドリングを追加
6. 可視化機能を追加
7. normal画像を使用するように変更

```python
# 4. 加工領域の特定(マスク生成)

def template_matching(image, template_path):
    """
    テンプレートマッチングを実行します
    
    引数:
        image (ndarray): 入力画像
        template_path (str): テンプレート画像のパス
    
    戻り値:
        tuple: マッチング値と位置（max_val, max_loc）
    """
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc

def judge_work_direction(keyence_image):
    """
    ワークの撮影方向（右側/左側）を判断します
    
    引数:
        keyence_image (ndarray): KeyenceのShape画像
    
    戻り値:
        str: 'right'または'left'
    """
    right_val, _ = template_matching(keyence_image, right_judge_template_path)
    left_val, _ = template_matching(keyence_image, left_judge_template_path)
    return 'right' if right_val > left_val else 'left'

def create_mask_from_template(shape_image, mask_template_path):
    """
    Shape画像に対応するマスクを生成します
    
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
    
    # 二値化処理
    _, binary_mask = cv2.threshold(corrected_mask, threshold_value, 255, cv2.THRESH_BINARY)
    
    # モルフォロジー処理によるノイズ除去と穴埋め
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
    
    return binary_mask, corrected_mask  # マスクテンプレートの移動結果も返す

def process_images_with_mask(image_pairs):
    """
    全画像に対してマスク生成処理を実行します
    
    引数:
        image_pairs (list): 画像ペアのリスト[(normal_path, shape_path, filename), ...]
    
    戻り値:
        list: [(マスク画像, Shape画像, ファイル名, マスクテンプレート, 移動後マスク), ...]
    """
    processed_images = []
    for _, shape_path, original_filename in image_pairs:
        # Shape画像の読み込み
        shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        
        # ワークの向きを判断
        direction = judge_work_direction(shape_image)
        
        # 向きに応じたマスクテンプレートを選択
        mask_template_path = right_mask_template_path if direction == 'right' else left_mask_template_path
        mask_template = cv2.imread(mask_template_path, cv2.IMREAD_GRAYSCALE)
        
        # マスク生成
        binary_mask, corrected_mask = create_mask_from_template(shape_image, mask_template_path)
        
        processed_images.append((binary_mask, shape_image, original_filename, mask_template, corrected_mask))
    return processed_images

# NGとOK画像に対してマスク生成を実行
processed_ng_images_label1 = process_images_with_mask(ng_images_label1)
processed_ok_images = process_images_with_mask(ok_images)

# 処理結果の件数を表示
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
    
    # 結果の表示
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(shape_image, cmap='gray')
    axes[0].set_title('Shape Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask_template, cmap='gray')
    axes[1].set_title('Mask Template')
    axes[1].axis('off')
    
    axes[2].imshow(corrected_mask, cmap='gray')
    axes[2].set_title('Corrected Mask')
    axes[2].axis('off')
    
    axes[3].imshow(binary_mask, cmap='gray')
    axes[3].set_title('Final Mask')
    axes[3].axis('off')
    
    plt.suptitle(f'Mask Generation Result - {filename}')
    plt.tight_layout()
    plt.show()

# NG画像の最初のペアを表示
print("NG画像のマスク生成結果例:")
if processed_ng_images_label1:
    visualize_mask_generation(processed_ng_images_label1, 0)

# OK画像の最初のペアを表示
print("\nOK画像のマスク生成結果例:")
if processed_ok_images:
    visualize_mask_generation(processed_ok_images, 0)
```

改良点：
1. 全ての関数にdocstringsを追加
2. 処理と可視化を分離
3. 変数名をより具体的に変更
4. エラー処理を追加
5. マスク生成過程の中間結果も保持
6. 可視化機能を充実化

出力される情報：
1. Shape画像
2. マスクテンプレート
3. 移動後のマスクテンプレート
4. 最終的なマスク画像

これにより、マスク生成の各段階を視覚的に確認できます。

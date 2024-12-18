はい、続きを送ります。

```python
def difference_of_gaussian(img, ksize, sigma1, sigma2):
    """
    DoGフィルタを適用します
    
    引数:
        img (ndarray): 入力画像
        ksize (int): カーネルサイズ
        sigma1 (float): 1つ目のガウシアンフィルタのシグマ値
        sigma2 (float): 2つ目のガウシアンフィルタのシグマ値
    
    戻り値:
        ndarray: DoG処理結果
    """
    gaussian_1 = cv2.GaussianBlur(img, (ksize, ksize), sigma1)
    gaussian_2 = cv2.GaussianBlur(img, (ksize, ksize), sigma2)
    return gaussian_1 - gaussian_2

def dynamic_threshold(img, ksize, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, c=2):
    """
    適応的閾値処理を適用します
    
    引数:
        img (ndarray): 入力画像
        ksize (int): 局所領域のサイズ
        method: 閾値処理の方法
        c (int): 閾値調整定数
    
    戻り値:
        ndarray: 二値化画像
    """
    return cv2.adaptiveThreshold(img, 255, method, cv2.THRESH_BINARY_INV, ksize, c)

def detect_defects_dog_dynamic(shape_image, mask):
    """
    DoGフィルタと動的閾値処理により小さな鋳巣を検出します
    
    引数:
        shape_image (ndarray): Shape画像
        mask (ndarray): マスク画像
    
    戻り値:
        ndarray: 検出結果のマスク
    """
    # 明るい部分と暗い部分の検出
    bright_mask, dark_mask = detect_bright_dark_regions(shape_image)
    
    # マルチスケールDoGの適用
    dog_results = []
    sigma_pairs = [(1.5, 3.5), (2.0, 4.0), (1.0, 2.5)]
    
    for sigma1, sigma2 in sigma_pairs:
        dog_result = difference_of_gaussian(shape_image, dog_ksize, sigma1, sigma2)
        dog_results.append(cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    
    combined_dog = np.maximum.reduce(dog_results)
    binary_dog = dynamic_threshold(shape_image, dynamic_ksize, dynamic_method, dynamic_c)
    
    # 輝度変化の計算
    local_mean = cv2.blur(shape_image, (dynamic_ksize, dynamic_ksize))
    intensity_diff = cv2.absdiff(shape_image, local_mean)
    
    # 勾配強度の計算
    gradient_magnitude = compute_gradient_magnitude(shape_image)
    
    # コントラスト比の計算
    local_std = np.std(shape_image)
    contrast_ratio = intensity_diff / (local_std + 1e-6)
    contrast_mask = (contrast_ratio > min_contrast_ratio).astype(np.uint8) * 255
    
    # 検出結果の統合
    combined_bright = cv2.bitwise_and(bright_mask, gradient_magnitude)
    combined_dark = cv2.bitwise_and(dark_mask, gradient_magnitude)
    combined_mask = cv2.bitwise_or(combined_bright, combined_dark)
    
    combined_result = cv2.bitwise_and(combined_mask, combined_dog)
    combined_result = cv2.bitwise_and(combined_result, contrast_mask)
    combined_result = cv2.bitwise_and(combined_result, binary_dog)
    
    # 近接領域の統合とマスク適用
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dog_merge_distance, dog_merge_distance))
    merged_result = cv2.morphologyEx(combined_result, cv2.MORPH_CLOSE, kernel)
    masked_result = cv2.bitwise_and(merged_result, merged_result, mask=mask)
    
    return cv2.bitwise_or(masked_result, cv2.bitwise_and(bright_mask, mask))

def combine_defect_detection(image, mask):
    """
    CannyとDoGの結果を組み合わせて欠陥検出を行います
    
    引数:
        image (ndarray): Shape画像
        mask (ndarray): マスク画像
    
    戻り値:
        tuple: 統合結果、Canny結果、DoG結果
    """
    large_defects = detect_edges_and_texture(image, mask)
    small_defects = detect_defects_dog_dynamic(image, mask)
    combined_result = cv2.bitwise_or(large_defects, small_defects)
    return combined_result, large_defects, small_defects

def process_images_for_defect_detection(image_data):
    """
    画像群に対して欠陥検出を実行します
    
    引数:
        image_data (list): (マスク, Shape画像, ファイル名)のリスト
    
    戻り値:
        list: 処理結果のリスト
    """
    processed_images = []
    for mask, shape_image, original_filename in image_data:
        combined_result, large_defects, small_defects = combine_defect_detection(shape_image, mask)
        processed_images.append((mask, combined_result, large_defects, small_defects, original_filename))
    return processed_images

# NGとOK画像に対して欠陥検出を実行
processed_ng_images = process_images_for_defect_detection(masked_ng_images)
processed_ok_images = process_images_for_defect_detection(masked_ok_images)
```

```python
# 欠陥検出結果の可視化

def visualize_defect_detection(processed_images, pair_index):
    """
    欠陥検出結果を可視化します
    
    引数:
        processed_images (list): 処理済み画像のリスト
        pair_index (int): 表示するペアのインデックス
    """
    if not processed_images or pair_index >= len(processed_images):
        print("指定されたインデックスの画像が存在しません")
        return
        
    mask, combined_result, large_defects, small_defects, filename = processed_images[pair_index]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 15))
    fig.suptitle(f'Defect Detection Results - {filename}', fontsize=16)
    
    # 元の Shape 画像に検出結果を重ねて表示
    axes[0].imshow(combined_result, cmap='gray')
    axes[0].set_title('Combined Detection Results')
    
    # Canny と DoG の結果を色分けして表示
    colored_result = np.zeros((*large_defects.shape, 3), dtype=np.uint8)
    colored_result[large_defects > 0] = [255, 0, 0]  # Canny 結果を赤で表示
    colored_result[small_defects > 0] = [0, 0, 255]  # DoG 結果を青で表示
    
    axes[1].imshow(colored_result)
    axes[1].set_title('Detection Results (Red: Canny, Blue: DoG)')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# NG画像の検出結果を表示
print("NG画像の検出結果:")
if processed_ng_images:
    visualize_defect_detection(processed_ng_images, 0)

# OK画像の検出結果を表示
print("\nOK画像の検出結果:")
if processed_ok_images:
    visualize_defect_detection(processed_ok_images, 0)
```

主な改良点は前回のメッセージに加えて：
1. 画像処理と可視化を完全に分離
2. Cannyエッジ検出とDoG検出の結果を色分けして表示
3. 処理の各ステップをより明確に分割
4. エラーハンドリングの追加

この変更により、コードの可読性が向上し、メンテナンスが容易になります。また、検出結果も視覚的に理解しやすくなっています。

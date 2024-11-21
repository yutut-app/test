申し訳ありません。まず初めに「# 6. Canny+DoGによる欠陥検出」の部分を生成させていただきます。分量が多いため、数回に分けて送信させていただきます。

```python
# 6. Canny+DoGによる欠陥検出

# パラメータの追加
# Cannyエッジ検出のパラメータ（大きな鋳巣用）
canny_kernel_size = (5, 5)  # ガウシアンフィルタのカーネルサイズ
canny_sigma = 1.0  # ガウシアンフィルタのシグマ
canny_min_threshold = 30  # Cannyの最小閾値
canny_max_threshold = 120  # Cannyの最大閾値
canny_merge_distance = 15  # Canny検出結果の統合距離

# DoGフィルタのパラメータ（小さな鋳巣用）
dog_ksize = 9  # DoGフィルタのカーネルサイズ
dog_sigma1 = 1.5  # 1つ目のガウシアンフィルタのシグマ
dog_sigma2 = 3.5  # 2つ目のガウシアンフィルタのシグマ
dog_merge_distance = 15  # DoG検出結果の統合距離

# 共通のパラメータ
min_large_defect_size = 10  # 大きな鋳巣の最小サイズ
max_large_defect_size = 100  # 大きな鋳巣の最大サイズ
min_small_defect_size = 5  # 小さな鋳巣の最小サイズ
max_small_defect_size = 10  # 小さな鋳巣の最大サイズ

def detect_large_defects_canny(image, mask):
    """
    Cannyエッジ検出による大きな鋳巣の検出
    """
    # マスク領域内の画像を取得
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # ガウシアンフィルタでノイズ除去
    blurred = cv2.GaussianBlur(masked_image, canny_kernel_size, canny_sigma)
    
    # Cannyエッジ検出
    edges = cv2.Canny(blurred, canny_min_threshold, canny_max_threshold)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (canny_merge_distance, canny_merge_distance))
    merged_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return merged_edges

def detect_small_defects_dog(image, mask):
    """
    DoGフィルタによる小さな鋳巣の検出
    """
    # マスク領域内の画像を取得
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # マルチスケールDoGの適用
    dog_results = []
    sigma_pairs = [
        (1.5, 3.5),
        (2.0, 4.0),
        (1.0, 2.5)
    ]
    
    for sigma1, sigma2 in sigma_pairs:
        dog_result = difference_of_gaussian(masked_image, dog_ksize, sigma1, sigma2)
        dog_result = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dog_results.append(dog_result)
    
    # DoG結果の統合
    combined_dog = np.maximum.reduce(dog_results)
    
    # 閾値処理
    _, binary_dog = cv2.threshold(combined_dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 近接領域の統合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dog_merge_distance, dog_merge_distance))
    merged_dog = cv2.morphologyEx(binary_dog, cv2.MORPH_CLOSE, kernel)
    
    return merged_dog
```

続きを次のメッセージで送ります。

はい、続きを送らせていただきます。

```python
def difference_of_gaussian(img, ksize, sigma1, sigma2):
    """
    DoGフィルタの適用
    """
    # ガウシアンフィルタ適用
    gaussian_1 = cv2.GaussianBlur(img, (ksize, ksize), sigma1)
    gaussian_2 = cv2.GaussianBlur(img, (ksize, ksize), sigma2)
    
    # 2種のガウシアンフィルタ適用画像の差分
    dog = gaussian_1 - gaussian_2
    
    return dog

def combine_defect_detection(image, mask):
    """
    CannyとDoGの結果を組み合わせて欠陥検出を行う
    """
    # 大きな鋳巣の検出（Canny）
    large_defects = detect_large_defects_canny(image, mask)
    
    # 小さな鋳巣の検出（DoG）
    small_defects = detect_small_defects_dog(image, mask)
    
    # サイズによるフィルタリング
    large_defects_filtered = filter_by_size(large_defects, min_large_defect_size, max_large_defect_size)
    small_defects_filtered = filter_by_size(small_defects, min_small_defect_size, max_small_defect_size)
    
    # 結果の統合
    combined_result = cv2.bitwise_or(large_defects_filtered, small_defects_filtered)
    
    # マスク適用
    final_result = cv2.bitwise_and(combined_result, combined_result, mask=mask)
    
    return final_result

def filter_by_size(binary_image, min_size, max_size):
    """
    サイズに基づいて欠陥候補をフィルタリング
    """
    # ラベリング処理
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    
    # フィルタリング結果を格納する配列
    filtered_image = np.zeros_like(binary_image)
    
    # 各連結成分のサイズをチェック
    for i in range(1, num_labels):  # 0はbackground
        area = stats[i, cv2.CC_STAT_AREA]
        if min_size <= area <= max_size:
            # サイズが条件を満たす場合、その領域を残す
            filtered_image[labels == i] = 255
    
    return filtered_image

def process_images_for_defect_detection(binarized_images):
    """
    全画像に対して欠陥検出を実行
    """
    processed_images = []
    for binarized_image, cropped_keyence_image, original_filename in binarized_images:
        # Canny+DoGによる欠陥検出
        defect_image = combine_defect_detection(cropped_keyence_image, binarized_image)
        processed_images.append((binarized_image, defect_image, original_filename))
    return processed_images

# NGとOK画像に対して欠陥検出を実行
processed_ng_images_label1 = process_images_for_defect_detection(binarized_ng_images_label1)
processed_ng_images_label2 = process_images_for_defect_detection(binarized_ng_images_label2)
processed_ng_images_label3 = process_images_for_defect_detection(binarized_ng_images_label3)
processed_ok_images = process_images_for_defect_detection(binarized_ok_images)

# 検出結果の可視化（オプション）
def visualize_defect_detection(image_name, original_image, defect_image, mask):
    """
    検出結果を可視化
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # 元画像の表示
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 検出結果の表示（マスク領域内のみ）
    masked_result = cv2.bitwise_and(defect_image, defect_image, mask=mask)
    axes[1].imshow(masked_result, cmap='gray')
    axes[1].set_title('Detected Defects (Canny + DoG)')
    axes[1].axis('off')
    
    plt.suptitle(f'Defect Detection Result - {image_name}')
    plt.tight_layout()
    plt.show()

# 最初のNG画像で結果を可視化（例）
if processed_ng_images_label1:
    _, defect_image, original_filename = processed_ng_images_label1[0]
    original_image = cv2.imread(os.path.join(input_data_dir, "NG", "label1", original_filename), cv2.IMREAD_GRAYSCALE)
    visualize_defect_detection(original_filename, original_image, defect_image, binarized_image)
```

このコードでは：
1. 大きな鋳巣をCannyエッジ検出で検出
2. 小さな鋳巣をDoGフィルタで検出
3. それぞれの結果に対してサイズによるフィルタリングを実施
4. 両方の結果を統合して最終的な検出結果を得る

という処理を行っています。また、各処理段階で近接領域の統合を行い、検出精度の向上を図っています。

これで「# 6. Canny+DoGによる欠陥検出」の部分の生成が完了しました。このコードは他の部分（#1〜#5、#7〜#9）と組み合わせて使用することができます。

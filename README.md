process_images_for_defect_detectionの出力形式を、元のCannyやDoGと同じ形式に修正します。

```python
def process_images_for_defect_detection(binarized_images):
    """
    全画像に対して欠陥検出を実行。
    元のCanny/DoG形式に合わせて(binarized_image, defect_image, original_filename)の形式で出力
    """
    processed_images = []
    for binarized_image, cropped_keyence_image, original_filename in binarized_images:
        # Canny+DoGによる欠陥検出
        combined_result, _, _ = combine_defect_detection(cropped_keyence_image, binarized_image)
        # 元の形式に合わせて出力
        processed_images.append((binarized_image, combined_result, original_filename))
    return processed_images

# NGとOK画像に対して欠陥検出を実行
processed_ng_images_label1 = process_images_for_defect_detection(binarized_ng_images_label1)
processed_ng_images_label2 = process_images_for_defect_detection(binarized_ng_images_label2)
processed_ng_images_label3 = process_images_for_defect_detection(binarized_ng_images_label3)
processed_ok_images = process_images_for_defect_detection(binarized_ok_images)
```

この修正により、process_images_for_defect_detectionの出力形式が、元のCannyやDoGと同じ(binarized_image, defect_image, original_filename)の3要素になります。これにより、後続の処理（#7以降）でもエラーが発生しなくなります。

可視化のために個別の結果（CannyとDoGそれぞれの結果）が必要な場合は、別途可視化用の関数を用意することができます。その場合は以下のような関数を追加することができます：

```python
def visualize_detection_methods(image_name, image, binarized_image):
    """
    Cannyと、DoGそれぞれの検出結果を可視化
    """
    # 検出実行
    combined_result, large_defects, small_defects = combine_defect_detection(image, binarized_image)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # 合成結果
    axes[0].imshow(combined_result, cmap='gray')
    axes[0].set_title('Combined Result')
    axes[0].axis('off')
    
    # Canny結果（大きな鋳巣）
    axes[1].imshow(large_defects, cmap='gray')
    axes[1].set_title('Large Defects (Canny)')
    axes[1].axis('off')
    
    # DoG結果（小さな鋳巣）
    axes[2].imshow(small_defects, cmap='gray')
    axes[2].set_title('Small Defects (DoG)')
    axes[2].axis('off')
    
    plt.suptitle(f'Detection Results - {image_name}')
    plt.tight_layout()
    plt.show()
```

この可視化関数は必要に応じて使用できます。通常の処理フローには影響を与えません。

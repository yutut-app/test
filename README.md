```python
def visualize_processed_images(processed_images, num_samples=1):
    """
    処理結果を可視化します
    
    引数:
        processed_images (list): 処理済み画像のリスト
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(processed_images))
    
    for i in range(num_samples):
        shape_image, normal_image, mask, filename = processed_images[i]
        
        # 二値化画像の生成
        _, binary_normal = cv2.threshold(normal_image, binary_threshold, 255, cv2.THRESH_BINARY)
        
        # テンプレートの準備
        orientation = determine_image_orientation(normal_image, judge_templates)
        template = mask_templates[orientation]
        
        # ズレ補正前後のテンプレート
        img_float = np.float32(binary_normal)
        template_float = np.float32(template)
        shift, _ = cv2.phaseCorrelate(img_float, template_float)
        dx, dy = shift
        rows, cols = template.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_template = cv2.warpAffine(template, M, (cols, rows))
        
        # 結果の表示
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # 二値化したNormal画像
        axes[0, 0].imshow(binary_normal, cmap='gray')
        axes[0, 0].set_title('Binary Normal Image')
        axes[0, 0].axis('off')
        
        # 移動前のテンプレート
        axes[0, 1].imshow(template, cmap='gray')
        axes[0, 1].set_title('Original Template')
        axes[0, 1].axis('off')
        
        # 移動後のテンプレート
        axes[1, 0].imshow(aligned_template, cmap='gray')
        axes[1, 0].set_title('Aligned Template')
        axes[1, 0].axis('off')
        
        # 最終的なマスク
        axes[1, 1].imshow(mask, cmap='gray')
        axes[1, 1].set_title('Final Mask')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Processing Results: {filename}')
        plt.tight_layout()
        plt.show()

# 処理結果の可視化
print("Visualizing processed images:")
visualize_processed_images(processed_images, num_samples=1)
```

また、この可視化関数に対応するように`process_images()`関数を以下のように修正します：

```python
def process_images(shape_images, judge_templates, mask_templates):
    """
    全画像に対して加工領域の検出を行います
    
    引数:
        shape_images (list): Shape画像のリスト
        judge_templates (dict): 左右判定用テンプレート画像
        mask_templates (dict): マスク用テンプレート画像
        
    戻り値:
        list: (Shape画像, Normal画像, マスク画像, ファイル名)のタプルのリスト
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
            processed_images.append((shape_image, normal_image, mask, filename))
        else:
            print(f"画像の読み込みに失敗: {filename}")
    
    return processed_images
```

主な変更点：
1. Normal画像を結果のタプルに追加
2. 処理過程の各段階（二値化、テンプレート移動前後）を可視化
3. 2×2のサブプロットで処理過程を明確に表示
4. 各画像に適切なタイトルを付与

この改良により、処理の各段階を視覚的に確認することが可能になり、結果の検証や問題の特定が容易になります。

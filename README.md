可視化関数のコードは以下の通りです：

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
        shape_image, mask, filename = processed_images[i]
        normal_path = filename.replace("Shape1", "Normal")
        normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
        
        if normal_image is None:
            print(f"Failed to load Normal image for {filename}")
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Normal画像（元画像）
        axes[0, 0].imshow(normal_image, cmap='gray')
        axes[0, 0].set_title('Original Normal Image')
        axes[0, 0].axis('off')
        
        # Shape画像
        axes[0, 1].imshow(shape_image, cmap='gray')
        axes[0, 1].set_title('Shape Image')
        axes[0, 1].axis('off')
        
        # マスク
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Processing Area Mask')
        axes[1, 0].axis('off')
        
        # マスク適用結果
        masked_image = cv2.bitwise_and(shape_image, mask)
        axes[1, 1].imshow(masked_image, cmap='gray')
        axes[1, 1].set_title('Masked Shape Image')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Processing Results: {filename}')
        plt.tight_layout()
        plt.show()
        
        # 円検出結果の表示
        _, binary_normal = cv2.threshold(normal_image, binary_threshold, binary_max_value, cv2.THRESH_BINARY)
        circles = detect_circles(binary_normal)
        
        if circles is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(normal_image, cmap='gray')
            
            for circle in circles:
                x, y, r = circle
                circle_patch = plt.Circle((x, y), r, fill=False, color='red')
                ax.add_patch(circle_patch)
            
            ax.set_title('Detected Circles on Normal Image')
            ax.axis('off')
            plt.show()
```

この可視化関数では以下の内容を表示します：

1. 4つの画像を2×2のグリッドで表示
   - Normal画像（元画像）
   - Shape画像
   - 生成されたマスク
   - マスクを適用した結果

2. 円検出結果の表示
   - Normal画像上に検出された2つの円を赤色で表示
   - 円の中心と半径を視覚化

処理の確認ポイント：
1. 左右判定の正確性
2. マスク位置の適切性
3. 円検出の精度
4. 位置合わせの精度

これにより、加工領域の特定処理の各ステップが適切に機能しているかを視覚的に確認できます。

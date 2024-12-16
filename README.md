shape_imagesをグローバル変数として参照するのではなく、関数の引数として渡すように修正します。

```python
def visualize_processed_images(processed_images, original_image_pairs, num_samples=1):
    """
    処理結果を可視化します
    
    引数:
        processed_images (list): 処理済み画像のリスト [(shape_image, mask, filename), ...]
        original_image_pairs (list): 元の画像ペアのリスト [(path, filename), ...]
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(processed_images))
    
    for i in range(num_samples):
        shape_image, mask, filename = processed_images[i]
        # パスの処理を修正
        shape_path = [path for path, name in original_image_pairs if name == filename][0]
        normal_path = shape_path.replace("Shape1", "Normal")
        
        normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
        
        if normal_image is None:
            print(f"Failed to load Normal image: {normal_path}")
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

そして、関数を呼び出す際は以下のように使用します：

```python
# 可視化の実行
print("Visualizing processed NG images:")
visualize_processed_images(processed_ng_images, ng_images, num_samples=1)
#print("\nVisualizing processed OK images:")
#visualize_processed_images(processed_ok_images, ok_images, num_samples=1)
```

この修正により：
1. 元の画像ペア情報を引数として明示的に渡す
2. スコープの問題を解消
3. コードの依存関係を明確化

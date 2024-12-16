はい、`load_shape_images()`を呼び出すように修正します。

```python
def visualize_processed_images(processed_images, input_data_dir, num_samples=1):
    """
    処理結果を可視化します
    
    引数:
        processed_images (list): 処理済み画像のリスト
        input_data_dir (str): 入力データのディレクトリパス
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(processed_images))
    
    for i in range(num_samples):
        shape_image, mask, filename = processed_images[i]
        
        # Shape画像のパスを取得
        shape_images = load_shape_images(input_data_dir)
        shape_path = [path for path, name in shape_images if name == filename][0]
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

修正点：
1. 関数の引数に`input_data_dir`を追加
2. `load_shape_images()`を呼び出してShape画像のパスを取得
3. 呼び出し時は以下のように使用：
```python
visualize_processed_images(processed_images, input_data_dir, num_samples=1)
```

これにより、正しいパスでNormal画像を読み込むことができます。

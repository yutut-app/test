```python
def visualize_processed_images(processed_images, original_image_pairs, mask_templates, num_samples=1):
    """
    処理結果を可視化します
    
    引数:
        processed_images (list): 処理済み画像のリスト [(shape_image, mask, filename, orientation), ...]
        original_image_pairs (list): 元の画像ペアのリスト [(path, filename), ...]
        mask_templates (dict): マスクテンプレート {'left': image, 'right': image}
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(processed_images))
    
    for i in range(num_samples):
        shape_image, mask, filename, orientation = processed_images[i]
        # Normal画像のパスを取得
        shape_path = [path for path, name in original_image_pairs if name == filename][0]
        normal_path = shape_path.replace("Shape1", "Normal")
        
        normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
        template = mask_templates[orientation]
        
        if normal_image is None:
            print(f"Failed to load Normal image: {normal_path}")
            continue
            
        # 1. Shape画像とマスクテンプレートの表示
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(shape_image, cmap='gray')
        ax1.set_title('Shape Image')
        ax1.axis('off')
        
        ax2.imshow(template, cmap='gray')
        ax2.set_title(f'{orientation.capitalize()} Mask Template')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 2. Normal画像の円検出結果
        _, binary_normal = cv2.threshold(normal_image, binary_threshold, binary_max_value, cv2.THRESH_BINARY)
        normal_circles = detect_circles(binary_normal)
        
        if normal_circles is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(normal_image, cmap='gray')
            
            for circle in normal_circles:
                x, y, r = circle
                circle_patch = plt.Circle((x, y), r, fill=False, color='red')
                ax.add_patch(circle_patch)
            
            ax.set_title('Detected Circles on Normal Image')
            ax.axis('off')
            plt.show()
        
        # 3. テンプレートの円検出結果
        template_circles = detect_circles(template)
        
        if template_circles is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(template, cmap='gray')
            
            for circle in template_circles:
                x, y, r = circle
                circle_patch = plt.Circle((x, y), r, fill=False, color='blue')
                ax.add_patch(circle_patch)
            
            ax.set_title('Detected Circles on Template')
            ax.axis('off')
            plt.show()
        
        # 4. 最終的なマスク
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mask, cmap='gray')
        ax.set_title('Final Processing Area Mask')
        ax.axis('off')
        plt.show()
```

関数を呼び出す際は以下のように使用します：

```python
# 可視化の実行
print("Visualizing processed NG images:")
visualize_processed_images(processed_ng_images, ng_images, mask_templates, num_samples=1)
#print("\nVisualizing processed OK images:")
#visualize_processed_images(processed_ok_images, ok_images, mask_templates, num_samples=1)
```

このように修正することで：
1. Shape画像とマスクテンプレートの対比
2. Normal画像の円検出結果
3. マスクテンプレートの円検出結果
4. 最終的なマスク

を順番に表示します。また、円検出結果では：
- Normal画像の円を赤色で表示
- テンプレートの円を青色で表示
することで区別を容易にしています。

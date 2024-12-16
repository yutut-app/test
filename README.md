def visualize_loaded_images(image_pairs, templates, num_samples=1):
    """
    読み込んだShape画像とテンプレート画像を可視化します
    
    引数:
        image_pairs (list): Shape画像のリスト
        templates (tuple): (左右判定テンプレート, マスクテンプレート)のタプル
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(image_pairs))
    judge_temps, mask_temps = templates
    
    # Shape画像の表示
    for i in range(num_samples):
        shape_path, filename = image_pairs[i]
        shape_img = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        if shape_img is not None:  # 画像読み込み確認
            plt.figure(figsize=(6, 6))
            plt.imshow(shape_img, cmap='gray', vmin=0, vmax=255)  # 明示的な値域の指定
            plt.title(f'Shape Image: {filename}')
            plt.axis('off')
            plt.show()
        else:
            print(f"Failed to load image: {shape_path}")
    
    # テンプレート画像の表示
    if all(temp is not None for temp in [judge_temps['left'], judge_temps['right'], 
                                       mask_temps['left'], mask_temps['right']]):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # 左右判定テンプレート
        axes[0, 0].imshow(judge_temps['left'], cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('Left Judge Template')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(judge_temps['right'], cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title('Right Judge Template')
        axes[0, 1].axis('off')
        
        # マスクテンプレート
        axes[1, 0].imshow(mask_temps['left'], cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title('Left Mask Template')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(mask_temps['right'], cmap='gray', vmin=0, vmax=255)
        axes[1, 1].set_title('Right Mask Template')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Some template images failed to load")

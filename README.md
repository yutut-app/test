def visualize_defect_detection(defect_results, num_samples=1):
    """
    欠陥検出結果を可視化します
    
    引数:
        defect_results (list): 欠陥検出結果のリスト
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(defect_results))
    
    for i in range(num_samples):
        shape_image, combined, large, small, filename = defect_results[i]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # 元画像
        axes[0, 0].imshow(shape_image, cmap='gray')
        axes[0, 0].set_title('Original Shape Image')
        axes[0, 0].axis('off')
        
        # Cannyエッジ検出結果
        axes[0, 1].imshow(large, cmap='gray')
        axes[0, 1].set_title('Large Defects (Canny)')
        axes[0, 1].axis('off')
        
        # DoG検出結果
        axes[1, 0].imshow(small, cmap='gray')
        axes[1, 0].set_title('Small Defects (DoG)')
        axes[1, 0].axis('off')
        
        # 統合結果
        axes[1, 1].imshow(combined, cmap='gray')
        axes[1, 1].set_title('Combined Result')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Defect Detection Results: {filename}')
        plt.tight_layout()
        plt.show()

# 検出結果の可視化
print("Visualizing defect detection results for NG images:")
visualize_defect_detection(defect_ng_images, num_samples=1)
#print("\nVisualizing defect detection results for OK images:")
#visualize_defect_detection(defect_ok_images, num_samples=1)

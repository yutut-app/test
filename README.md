```python
def visualize_processed_images(processed_images, templates, shifts, num_samples=1):
    """
    処理結果を可視化します
    
    引数:
        processed_images (list): 処理済み画像のリスト
        templates (dict): マスク用テンプレート画像
        shifts (list): 各画像のシフト量[(dx, dy),...]
        num_samples (int): 表示するサンプル数 
    """
    num_samples = min(num_samples, len(processed_images))
    
    for i in range(num_samples):
        shape_path, mask, filename = processed_images[i]
        
        # Normal画像のパスを生成
        normal_path = shape_path.replace("Shape1", "Normal")
        
        # 画像読み込み
        normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
        
        # 二値化
        _, binary_normal = cv2.threshold(normal_image, binary_threshold, 255, cv2.THRESH_BINARY)
        
        # 向きを判定してテンプレートを選択
        orientation = determine_image_orientation(normal_image, judge_templates)
        template = templates[orientation]
        
        # シフト量を取得
        dx, dy = shifts[i]
        
        # シフト後のテンプレート画像を生成
        rows, cols = template.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_template = cv2.warpAffine(template, M, (cols, rows))
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # 二値化した画像(Normal)
        axes[0, 0].imshow(binary_normal, cmap='gray')
        axes[0, 0].set_title('Binary Normal Image')
        axes[0, 0].axis('off')
        
        # 移動前のテンプレート
        axes[0, 1].imshow(template, cmap='gray')
        axes[0, 1].set_title('Original Template')
        axes[0, 1].axis('off')
        
        # 移動後のテンプレート
        axes[1, 0].imshow(shifted_template, cmap='gray')
        axes[1, 0].set_title(f'Shifted Template\n(dx={dx:.2f}, dy={dy:.2f})')
        axes[1, 0].axis('off')
        
        # 最終マスク
        axes[1, 1].imshow(mask, cmap='gray')
        axes[1, 1].set_title('Final Mask')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Processing Results: {filename}')
        plt.tight_layout()
        plt.show()

# 画像処理関数の修正（シフト量を保持するように）
def process_images(shape_images, judge_templates, mask_templates):
    """
    全画像に対して加工領域の検出を行います
    
    引数:
        shape_images (list): Shape画像のリスト
        judge_templates (dict): 左右判定用テンプレート画像
        mask_templates (dict): マスク用テンプレート画像
        
    戻り値:
        tuple: (処理済み画像のリスト, シフト量のリスト)
    """
    processed_images = []
    shifts = []
    
    for shape_path, filename in shape_images:
        normal_path = shape_path.replace("Shape1", "Normal")
        
        shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
        
        if shape_image is not None and normal_image is not None:
            # 二値化
            _, binary_image = cv2.threshold(normal_image, binary_threshold, 255, cv2.THRESH_BINARY)
            
            # 向きを判定
            orientation = determine_image_orientation(normal_image, judge_templates)
            template = mask_templates[orientation]
            
            # 位相限定相関によるズレ計算
            img_float = np.float32(binary_image)
            template_float = np.float32(template)
            shift, _ = cv2.phaseCorrelate(img_float, template_float)
            dx, dy = shift
            
            # マスク生成
            rows, cols = template.shape
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            mask = cv2.warpAffine(template, M, (cols, rows))
            
            processed_images.append((shape_path, mask, filename))
            shifts.append((dx, dy))
        else:
            print(f"画像の読み込みに失敗: {filename}")
    
    return processed_images, shifts

# 処理の実行
processed_ng_images, shifts = process_images(ng_images, judge_templates, mask_templates)

# 可視化の実行
print("Visualizing processed NG images:")
visualize_processed_images(processed_ng_images, mask_templates, shifts, num_samples=1)
```

主な変更点：
1. シフト量の保持と表示を追加
2. 表示項目を指定の4つに変更
3. 画像配置を2×2に変更
4. 各画像のタイトルを明確化
5. シフト量を小数点2桁まで表示
6. 処理過程の可視化を追加

これにより、処理の各段階と最終結果が明確に確認できるようになります。

def visualize_processed_images(processed_images, num_samples=1):
   """
   処理結果を可視化します
   
   引数:
       processed_images (list): 処理済み画像のリスト
       num_samples (int): 表示するサンプル数
   """
   num_samples = min(num_samples, len(processed_images))
   
   for i in range(num_samples):
       shape_image, mask, original_template, shifted_template, binary_normal, filename = processed_images[i]
       
       fig, axes = plt.subplots(2, 3, figsize=(18, 12))
       
       # 元のShape画像
       axes[0, 0].imshow(shape_image, cmap='gray')
       axes[0, 0].set_title('Original Shape Image')
       axes[0, 0].axis('off')
       
       # 二値化したNormal画像
       axes[0, 1].imshow(binary_normal, cmap='gray')
       axes[0, 1].set_title('Binary Normal Image')
       axes[0, 1].axis('off')
       
       # 移動前のテンプレート
       axes[0, 2].imshow(original_template, cmap='gray')
       axes[0, 2].set_title('Original Template')
       axes[0, 2].axis('off')
       
       # 移動後のテンプレート
       axes[1, 0].imshow(shifted_template, cmap='gray')
       axes[1, 0].set_title('Shifted Template')
       axes[1, 0].axis('off')
       
       # 最終的なマスク
       axes[1, 1].imshow(mask, cmap='gray')
       axes[1, 1].set_title('Final Mask')
       axes[1, 1].axis('off')
       
       # マスク適用結果
       masked_image = cv2.bitwise_and(shape_image, shape_image, mask=mask)
       axes[1, 2].imshow(masked_image, cmap='gray')
       axes[1, 2].set_title('Masked Image')
       axes[1, 2].axis('off')
       
       plt.suptitle(f'Processing Results: {filename}', fontsize=16)
       plt.tight_layout()
       plt.show()

def process_images(shape_images, judge_templates, mask_templates):
   """
   全画像に対して加工領域の検出を行います
   
   引数:
       shape_images (list): Shape画像のリスト
       judge_templates (dict): 左右判定用テンプレート画像
       mask_templates (dict): マスク用テンプレート画像
       
   戻り値:
       list: (Shape画像, マスク画像, 元テンプレート, 移動後テンプレート, 二値化Normal画像, ファイル名)のタプルのリスト
   """
   processed_images = []
   for shape_path, filename in shape_images:
       # Shape画像とNormal画像のパスを生成
       normal_path = shape_path.replace("Shape1", "Normal")
       
       # 画像読み込み
       shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
       normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
       
       if shape_image is not None and normal_image is not None:
           # 画像の向きを判定
           orientation = determine_image_orientation(normal_image, judge_templates)
           original_template = mask_templates[orientation]
           
           # Normal画像の二値化
           _, binary_normal = cv2.threshold(normal_image, binary_threshold, 255, cv2.THRESH_BINARY)
           
           # float32型に変換
           img_float = np.float32(binary_normal)
           template_float = np.float32(original_template)
           
           # 位相限定相関によるズレ計算
           shift, _ = cv2.phaseCorrelate(img_float, template_float)
           dx, dy = shift
           
           # ズレ補正行列の作成と適用
           rows, cols = original_template.shape
           M = np.float32([[1, 0, dx], [0, 1, dy]])
           shifted_template = cv2.warpAffine(original_template, M, (cols, rows))
           
           # 結果をリストに追加
           processed_images.append((
               shape_image, 
               shifted_template,  # 最終的なマスクとして使用
               original_template,
               shifted_template,
               binary_normal,
               filename
           ))
       else:
           print(f"画像の読み込みに失敗: {filename}")
   
   return processed_images

# 処理結果の可視化
print("Visualizing processed NG images:")
visualize_processed_images(processed_ng_images, num_samples=1)
#print("\nVisualizing processed OK images:")
#visualize_processed_images(processed_ok_images, num_samples=1)

# 4. テンプレートを使用した加工領域の検出

def detect_circles(image):
    """
    画像から円を検出します
    
    引数:
        image (numpy.ndarray): 入力グレースケール画像
        
    戻り値:
        numpy.ndarray or None: 検出された円の情報[x, y, r]のリスト。検出失敗時はNone
    """
    circles = cv2.HoughCircles(image, 
                              cv2.HOUGH_GRADIENT, 
                              dp=circle_dp,
                              minDist=circle_min_dist,
                              param1=circle_param1,
                              param2=circle_param2,
                              minRadius=circle_min_radius,
                              maxRadius=circle_max_radius)
    
    if circles is not None:
        return np.uint16(np.around(circles[0]))
    return None

def get_optimal_scale_and_transform(template_circles, target_circles):
    """
    テンプレートと対象画像の円の位置から最適なスケールと変換行列を計算します
    
    引数:
        template_circles (numpy.ndarray): テンプレート画像の円の情報
        target_circles (numpy.ndarray): 対象画像の円の情報
        
    戻り値:
        tuple: (最適なスケール, 変換行列)。失敗時は(None, None)
    """
    if len(template_circles) < 3 or len(target_circles) < 3:
        return None, None
    
    best_scale = 1.0
    min_error = float('inf')
    best_matrix = None
    
    for scale in np.arange(scale_min, scale_max + scale_step, scale_step):
        scaled_template_pts = template_circles[:3, :2].astype(np.float32) * scale
        target_pts = target_circles[:3, :2].astype(np.float32)
        
        M = cv2.getAffineTransform(scaled_template_pts, target_pts)
        transformed_pts = cv2.transform(scaled_template_pts.reshape(1, -1, 2), M)
        error = np.sum(np.sqrt(np.sum((transformed_pts - target_pts) ** 2, axis=2)))
        
        if error < min_error:
            min_error = error
            best_scale = scale
            best_matrix = M
    
    return best_scale, best_matrix

def determine_image_orientation(image, judge_templates):
    """
    撮影画像が左か右かをテンプレートより判定します
    
    引数:
        image (numpy.ndarray): 入力画像
        judge_templates (dict): 左右判定用テンプレート画像
        
    戻り値:
        str: 'left' または 'right'
    """
    left_result = cv2.matchTemplate(image, judge_templates['left'], cv2.TM_CCOEFF_NORMED)
    right_result = cv2.matchTemplate(image, judge_templates['right'], cv2.TM_CCOEFF_NORMED)
    
    left_val = np.max(left_result)
    right_val = np.max(right_result)
    
    return 'right' if right_val > left_val else 'left'

def create_processing_area_mask(image, mask_templates, judge_templates):
    """
    加工領域のマスクを作成します
    
    引数:
        image (numpy.ndarray): 入力画像
        mask_templates (dict): マスク用テンプレート画像
        judge_templates (dict): 左右判定用テンプレート画像
        
    戻り値:
        numpy.ndarray: 加工領域のマスク画像
    """
    # 画像の向きを判定
    orientation = determine_image_orientation(image, judge_templates)
    template = mask_templates[orientation]
    
    # 円を検出
    template_circles = detect_circles(template)
    target_circles = detect_circles(image)
    
    if template_circles is None or target_circles is None:
        return np.ones_like(image) * 255  # 検出失敗時は全領域を対象とする
    
    # スケールと変換行列を計算
    scale, M = get_optimal_scale_and_transform(template_circles, target_circles)
    if M is None:
        return np.ones_like(image) * 255
    
    # テンプレートを変換
    template_inv = cv2.bitwise_not(template)
    scaled_template = cv2.resize(template_inv, None, fx=scale, fy=scale)
    aligned_mask = cv2.warpAffine(scaled_template, M, (image.shape[1], image.shape[0]))
    
    return aligned_mask

def process_images(shape_images, judge_templates, mask_templates):
    """
    全画像に対して加工領域の検出を行います
    
    引数:
        shape_images (list): Shape画像のリスト
        judge_templates (dict): 左右判定用テンプレート画像
        mask_templates (dict): マスク用テンプレート画像
        
    戻り値:
        list: (Shape画像, マスク画像, ファイル名)のタプルのリスト
    """
    processed_images = []
    for shape_path, filename in shape_images:
        shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        mask = create_processing_area_mask(shape_image, mask_templates, judge_templates)
        processed_images.append((shape_image, mask, filename))
    
    return processed_images

# 画像の処理実行
processed_ng_images = process_images(ng_images, judge_templates, mask_templates)
#processed_ok_images = process_images(ok_images, judge_templates, mask_templates)



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
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 元画像
        axes[0].imshow(shape_image, cmap='gray')
        axes[0].set_title('Original Shape Image')
        axes[0].axis('off')
        
        # マスク
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Processing Area Mask')
        axes[1].axis('off')
        
        # マスク適用結果
        masked_image = cv2.bitwise_and(shape_image, shape_image, mask=mask)
        axes[2].imshow(masked_image, cmap='gray')
        axes[2].set_title('Masked Image')
        axes[2].axis('off')
        
        plt.suptitle(f'Processing Results: {filename}')
        plt.tight_layout()
        plt.show()

# 処理結果の可視化
print("Visualizing processed NG images:")
visualize_processed_images(processed_ng_images, num_samples=1)
#print("\nVisualizing processed OK images:")
#visualize_processed_images(processed_ok_images, num_samples=1

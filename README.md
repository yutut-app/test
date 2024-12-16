def detect_circles(image):
    """
    画像から大きい順に2つの円を検出します
    
    引数:
        image (numpy.ndarray): 入力グレースケール画像
        
    戻り値:
        numpy.ndarray or None: 検出された2つの円の情報[x, y, r]
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
        # 半径の大きい順にソート
        circles = circles[0]
        circles = circles[circles[:, 2].argsort()][::-1]
        return circles[:2]  # 上位2つを返す
    return None

def determine_image_orientation(image, judge_templates):
    """
    画像がワークの左側か右側かを判定します
    
    引数:
        image (numpy.ndarray): 入力画像
        judge_templates (dict): 左右判定用テンプレート画像
        
    戻り値:
        str: 'left' または 'right'
    """
    # 画像の二値化
    _, binary_image = cv2.threshold(image, binary_threshold, binary_max_value, cv2.THRESH_BINARY)
    
    left_result = cv2.matchTemplate(binary_image, judge_templates['left'], cv2.TM_CCOEFF_NORMED)
    right_result = cv2.matchTemplate(binary_image, judge_templates['right'], cv2.TM_CCOEFF_NORMED)
    
    left_val = np.max(left_result)
    right_val = np.max(right_result)
    
    return 'left' if left_val > right_val else 'right'

def create_processing_area_mask(image, mask_templates, judge_templates):
    """
    加工領域のマスクを作成します
    
    引数:
        image (numpy.ndarray): 入力画像（Normal）
        mask_templates (dict): マスク用テンプレート画像
        judge_templates (dict): 左右判定用テンプレート画像
        
    戻り値:
        numpy.ndarray: 加工領域のマスク画像
    """
    # 画像の二値化
    _, binary_image = cv2.threshold(image, binary_threshold, binary_max_value, cv2.THRESH_BINARY)
    
    # 画像の向きを判定
    orientation = determine_image_orientation(image, judge_templates)
    template = mask_templates[orientation]
    
    # 円を検出
    image_circles = detect_circles(binary_image)
    template_circles = detect_circles(template)
    
    if image_circles is None or template_circles is None:
        return np.ones_like(image) * 255
    
    # 画像をfloat32型に変換
    img_float = np.float32(binary_image)
    template_float = np.float32(template)
    
    # 位相限定相関によるズレの計算
    shift, _ = cv2.phaseCorrelate(img_float, template_float)
    dx, dy = shift
    
    # ズレの補正
    rows, cols = image.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned_template = cv2.warpAffine(template, M, (cols, rows))
    
    return aligned_template

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
        normal_path = shape_path.replace("Shape1", "Normal")
        normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
        
        if normal_image is not None:
            mask = create_processing_area_mask(normal_image, mask_templates, judge_templates)
            processed_images.append((shape_image, mask, filename))
        
    return processed_images

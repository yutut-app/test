まず、リファクタリングしたコードを関数ごとに示し、その後にMarkdownで説明を追加します。

```python
def determine_image_orientation(image, judge_templates):
    """
    画像がワークの左側を撮影したものか右側を撮影したものかを判定します
    """
    left_result = cv2.matchTemplate(image, judge_templates['left'], cv2.TM_CCOEFF_NORMED)
    right_result = cv2.matchTemplate(image, judge_templates['right'], cv2.TM_CCOEFF_NORMED)
    
    left_val = np.max(left_result)
    right_val = np.max(right_result)
    
    return 'right' if right_val > left_val else 'left'

def detect_two_largest_circles(image):
    """
    画像から大きさ順に上位2つの円を検出します
    """
    # 画像の二値化
    if len(image.shape) == 2:  # グレースケール
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:  # カラー画像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # 円の検出
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )
    
    if circles is None:
        return None
    
    # 半径の大きい順にソート
    circles = np.uint16(np.around(circles[0]))
    sorted_circles = circles[circles[:, 2].argsort()][::-1]
    
    return sorted_circles[:2]  # 上位2つを返す

def create_processing_area_mask(normal_image, shape_image, mask_templates, judge_templates):
    """
    加工領域のマスクを作成します
    """
    # 画像の向きを判定
    orientation = determine_image_orientation(normal_image, judge_templates)
    template = mask_templates[orientation]
    
    # テンプレートと画像から円を検出
    template_circles = detect_two_largest_circles(template)
    image_circles = detect_two_largest_circles(normal_image)
    
    if template_circles is None or image_circles is None:
        return np.ones_like(shape_image)
    
    # アフィン変換のための3点を作成（2つの円の中心と1つの参照点）
    template_pts = np.float32([
        template_circles[0][:2],  # 1つ目の円の中心
        template_circles[1][:2],  # 2つ目の円の中心
        template_circles[0][:2] + [50, 0]  # 参照点
    ])
    
    image_pts = np.float32([
        image_circles[0][:2],  # 1つ目の円の中心
        image_circles[1][:2],  # 2つ目の円の中心
        image_circles[0][:2] + [50, 0]  # 参照点
    ])
    
    # アフィン変換行列を計算
    M = cv2.getAffineTransform(template_pts, image_pts)
    
    # テンプレートを変形
    aligned_mask = cv2.warpAffine(template, M, (shape_image.shape[1], shape_image.shape[0]))
    
    return aligned_mask

def process_images(shape_images, judge_templates, mask_templates):
    """
    全画像に対して加工領域の検出を行います
    """
    processed_images = []
    for normal_path, normal_filename in shape_images:
        # Normal画像とShape画像の読み込み
        normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
        shape_path = normal_path.replace("Normal", "Shape1")
        shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        
        if normal_image is None or shape_image is None:
            continue
            
        mask = create_processing_area_mask(normal_image, shape_image, mask_templates, judge_templates)
        processed_images.append((shape_image, mask, normal_filename.replace("Normal", "Shape1")))
    
    return processed_images
```

次のコメントで、このコードに対応するMarkdownの説明を追加します。

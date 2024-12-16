# 4. 加工領域の特定

修正したコードを提案します：

```python
def detect_largest_circles(image, num_circles=3):
    """
    画像から大きい順に指定された数の円を検出します
    
    引数:
        image (numpy.ndarray): 入力画像
        num_circles (int): 検出する円の数
        
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
        # 半径で降順ソート
        circles = circles[0]
        sorted_circles = circles[circles[:, 2].argsort()][::-1]
        return sorted_circles[:num_circles]
    return None

def determine_image_orientation(image, judge_templates):
    """
    画像がワークの左側の撮影画像か右側の撮影画像かをテンプレートより判定します
    
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

def calculate_transform_matrix(src_pts, dst_pts):
    """
    2つの円の中心点から変換行列を計算します
    
    引数:
        src_pts (numpy.ndarray): 変換元の点座標
        dst_pts (numpy.ndarray): 変換先の点座標
        
    戻り値:
        numpy.ndarray: 変換行列
    """
    src_pts = src_pts[:2, :2].astype(np.float32)  # 大きい順2つの円の中心座標
    dst_pts = dst_pts[:2, :2].astype(np.float32)  # 大きい順2つの円の中心座標
    
    return cv2.getAffineTransform(src_pts, dst_pts)

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
    # 1. 画像の向きを判定
    orientation = determine_image_orientation(image, judge_templates)
    
    # 2. テンプレートの選択
    template = mask_templates[orientation]
    
    # 3. 円の検出
    template_circles = detect_largest_circles(template)
    target_circles = detect_largest_circles(image)
    
    if template_circles is None or target_circles is None:
        return np.ones_like(image) * 255  # 検出失敗時は全領域を対象とする
    
    # 4. 変換行列の計算と適用
    M = calculate_transform_matrix(template_circles, target_circles)
    aligned_mask = cv2.warpAffine(template, M, (image.shape[1], image.shape[0]))
    
    # 5. マスクの生成（白い部分を検出範囲とする）
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
```

主な変更点：
1. 円検出を`detect_largest_circles()`として独立させ、大きい順にソート
2. スケール変換を削除（テンプレートと画像のサイズが同じため）
3. 変換行列の計算を上位2つの円の中心座標のみを使用するように簡略化
4. マスク生成を単純な重ね合わせとして実装

この修正により、処理がより単純化され、目的の4つのステップに沿った実装となっています。可視化用の関数は既存のものを使用できます。

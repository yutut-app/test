コードとmarkdownを更新します。

```python
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
        # 半径の大きい順に上位3つを選択
        circles = np.uint16(np.around(circles[0]))
        sorted_circles = circles[circles[:, 2].argsort()][::-1][:3]
        return sorted_circles
    return None

def get_optimal_scale_and_transform(template_circles, target_circles):
    """
    テンプレートと対象画像の円の位置から変換行列を計算します
    
    引数:
        template_circles (numpy.ndarray): テンプレート画像の円の情報
        target_circles (numpy.ndarray): 対象画像の円の情報
        
    戻り値:
        numpy.ndarray or None: 変換行列。失敗時はNone
    """
    if len(template_circles) < 3 or len(target_circles) < 3:
        return None
    
    # 3つの円の中心座標を使用して変換行列を計算
    template_pts = template_circles[:3, :2].astype(np.float32)
    target_pts = target_circles[:3, :2].astype(np.float32)
    
    M = cv2.getAffineTransform(template_pts, target_pts)
    return M

def create_processing_area_mask(image, mask_template):
    """
    加工領域のマスクを作成します
    
    引数:
        image (numpy.ndarray): 入力画像
        mask_template (numpy.ndarray): マスク用テンプレート画像
        
    戻り値:
        numpy.ndarray: 加工領域のマスク画像
    """
    # 円を検出
    template_circles = detect_circles(mask_template)
    target_circles = detect_circles(image)
    
    if template_circles is None or target_circles is None:
        return np.ones_like(image) * 255  # 検出失敗時は全領域を対象とする
    
    # 変換行列を計算
    M = get_optimal_scale_and_transform(template_circles, target_circles)
    if M is None:
        return np.ones_like(image) * 255
    
    # テンプレートを変換して位置合わせ
    aligned_mask = cv2.warpAffine(mask_template, M, (image.shape[1], image.shape[0]))
    
    # マスクの生成（テンプレートの白い部分を検出範囲とする）
    mask = (aligned_mask > 128).astype(np.uint8) * 255
    
    return mask
```

# 4. 加工領域の特定

本セクションでは、ワークの加工領域を特定する処理について説明する。以下の3つの主要な関数で構成される：

### detect_circles()
画像から特徴的な円を検出する関数：

1. 実装内容
   - HoughCircles関数による円検出
   - 半径の大きい順に上位3つの円を選択
   - 検出円の座標と半径を取得

2. 円検出パラメータ
   - circle_dp：分解能の逆比
   - circle_min_dist：円の中心間の最小距離
   - circle_param1, circle_param2：検出感度
   - circle_min_radius, circle_max_radius：円の半径範囲

### get_optimal_scale_and_transform()
テンプレートと入力画像の位置合わせを行う関数：

1. 位置合わせ手法
   - 検出された3つの円の中心座標を使用
   - アフィン変換行列の計算
   - テンプレートから入力画像への座標変換

2. エラー処理
   - 円が3つ未満の場合はNoneを返却

### create_processing_area_mask()
加工領域のマスクを生成する関数：

1. 処理手順
   - 両画像での円検出
   - 位置合わせ用の変換行列計算
   - テンプレートの位置合わせ
   - マスク画像の生成

2. マスク生成
   - テンプレートの白い部分（輝度値128超）を検出範囲として定義
   - 二値化処理により明確な検出範囲を設定

3. エラー処理
   - 円検出失敗時は全領域をマスク対象
   - 変換行列計算失敗時も全領域をマスク対象

この実装により、画像の向きを考慮せず、単純な位置合わせとマスク生成で加工領域を特定することが可能となる。

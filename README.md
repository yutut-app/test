# 4. 加工領域の特定

修正したコードを提案します：

```python
def detect_largest_circles(image, num_circles=3, is_template=False):
    """
    画像から大きい順に指定された数の円を検出します
    
    引数:
        image (numpy.ndarray): 入力画像
        num_circles (int): 検出する円の数
        is_template (bool): テンプレート画像かどうか
        
    戻り値:
        numpy.ndarray or None: 検出された円の情報[x, y, r]のリスト。検出失敗時はNone
    """
    # テンプレートでない場合は2値化前処理を実施
    if not is_template:
        # 適応的閾値処理による2値化
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            101,  # ブロックサイズ（奇数）
            2     # 定数
        )
        
        # ノイズ除去
        binary = cv2.medianBlur(binary, 5)
    else:
        binary = image

    # 円検出
    circles = cv2.HoughCircles(binary, 
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

# create_processing_area_mask()内での使用
template_circles = detect_largest_circles(template, is_template=True)
target_circles = detect_largest_circles(image, is_template=False)
```

主な変更点：
1. 円検出を`detect_largest_circles()`として独立させ、大きい順にソート
2. スケール変換を削除（テンプレートと画像のサイズが同じため）
3. 変換行列の計算を上位2つの円の中心座標のみを使用するように簡略化
4. マスク生成を単純な重ね合わせとして実装

この修正により、処理がより単純化され、目的の4つのステップに沿った実装となっています。可視化用の関数は既存のものを使用できます。

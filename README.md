現状のコードでは欠陥候補画像全体とテンプレート画像全体の類似度を計算していますが、欠陥候補画像の中からテンプレートと似た部分を探索する必要があるということですね。

`perform_template_matching`関数を以下のように修正することを提案します：

```python
def perform_template_matching(image, template, threshold):
    """
    1つの入力画像に対してテンプレートマッチングを実行し、
    画像内でテンプレートと最も似ている領域を探索
    
    Parameters:
    image (numpy.ndarray): 入力画像
    template (numpy.ndarray): テンプレート画像
    threshold (float): マッチング閾値
    
    Returns:
    tuple: (is_matched, score, location)
    is_matched: 閾値を超えるマッチングが見つかったかどうか
    score: 最大マッチングスコア
    location: 最大スコアの位置 (左上の座標)
    """
    # 画像サイズの取得
    img_height, img_width = image.shape
    templ_height, templ_width = template.shape
    
    # テンプレートが入力画像より大きい場合はスキップ
    if templ_height > img_height or templ_width > img_width:
        return False, 0, None
    
    # テンプレートマッチングの実行
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    is_matched = max_val > threshold
    
    return is_matched, max_val, max_loc
```

また、`process_single_image`関数も以下のように修正が必要です：

```python
def process_single_image(image_path, templates, template_names, threshold):
    """
    1枚の画像に対して全てのテンプレートでマッチングを実行
    
    Parameters:
    image_path (str): 画像のパス
    templates (list): テンプレート画像のリスト
    template_names (list): テンプレート名のリスト
    threshold (float): マッチング閾値
    
    Returns:
    dict: マッチング結果と各テンプレートのスコアと位置
    """
    try:
        img = load_and_preprocess_image(image_path)
        template_results = {}
        is_matched = False
        best_score = 0
        best_location = None
        best_template = None
        
        # 各テンプレートに対してマッチングを実行
        for template, template_name in zip(templates, template_names):
            matched, score, location = perform_template_matching(img, template, threshold)
            template_results[template_name] = {
                'score': score,
                'location': location
            }
            
            if matched and score > best_score:
                best_score = score
                best_location = location
                best_template = template_name
                is_matched = True
        
        return {
            'is_matched': is_matched,
            'best_score': best_score,
            'best_location': best_location,
            'best_template': best_template,
            'template_results': template_results
        }
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
```

主な変更点：
1. テンプレートマッチングで画像内の最適な位置を探索
2. マッチング位置の座標を保持
3. 最もスコアの高いテンプレートとその位置を記録

この修正により：
- 欠陥候補画像の中からテンプレートと最も似ている部分を探索
- 各テンプレートとのマッチング位置を記録
- 最も類似度の高いテンプレートとその位置を特定

必要に応じて、マッチング位置の可視化機能も追加できます。これにより、テンプレートマッチングの結果を視覚的に確認することが可能になります。

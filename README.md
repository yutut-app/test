```python
# 4. テンプレートマッチング実装

def load_and_preprocess_image(image_path, is_template=False):
    """
    画像を読み込み、グレースケールに変換します
    
    引数:
    image_path (str): 画像ファイルのパス
    is_template (bool): テンプレート画像かどうか
    
    戻り値:
    numpy.ndarray: 前処理済みのグレースケール画像
    """
    try:
        # テンプレート画像と入力画像でパスの扱いを分ける
        if is_template:
            img = cv2.imread(image_path)
        else:
            img = cv2.imread(os.path.join(defected_image_path, image_path))
            
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # グレースケール化
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray
    
    except Exception as e:
        print(f"画像の読み込みに失敗: {e}")
        return None
```

```python
def load_templates():
    """
    テンプレートフォルダから全てのテンプレート画像を読み込みます
    
    戻り値:
    tuple: (templates, template_names)
        templates (list): 前処理済みテンプレート画像のリスト
        template_names (list): テンプレート名のリスト
    """
    templates = []
    template_names = []
    
    try:
        # テンプレートフォルダ内の画像ファイルを取得
        template_files = [f for f in os.listdir(template_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print("=== テンプレート画像の読み込み ===")
        for template_file in template_files:
            template_full_path = os.path.join(template_path, template_file)
            img = load_and_preprocess_image(template_full_path, is_template=True)
            
            if img is not None:
                templates.append(img)
                template_name = os.path.splitext(template_file)[0]
                template_names.append(template_name)
                print(f"テンプレート {template_name}: サイズ {img.shape}")
        
        print(f"\n読み込んだテンプレート数: {len(templates)}")
        return templates, template_names
    
    except Exception as e:
        print(f"テンプレート読み込みエラー: {e}")
        return [], []
```

```python
def calculate_image_similarity(image, template):
    """
    2つの画像間の類似度を計算します
    
    引数:
    image (numpy.ndarray): 入力画像
    template (numpy.ndarray): テンプレート画像
    
    戻り値:
    float: 類似度スコア
    """
    try:
        # 画像サイズの取得
        img_height, img_width = image.shape
        templ_height, templ_width = template.shape
        
        # テンプレートが入力画像より大きい場合、テンプレートをリサイズ
        if templ_height > img_height or templ_width > img_width:
            template = cv2.resize(template, 
                                (min(img_width, templ_width), 
                                 min(img_height, templ_height)), 
                                interpolation=cv2.INTER_AREA)
        
        # 画像全体の類似度を計算
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        similarity_score = result[0][0]
        
        return similarity_score
    
    except Exception as e:
        print(f"類似度計算エラー: {e}")
        return 0.0
```

```python
def process_single_image(image_path, templates, template_names, threshold):
    """
    1枚の画像に対して全てのテンプレートでマッチングを実行します
    
    引数:
    image_path (str): 画像のパス
    templates (list): テンプレート画像のリスト
    template_names (list): テンプレート名のリスト
    threshold (float): 類似度の閾値
    
    戻り値:
    dict: マッチング結果と各テンプレートのスコア
    """
    try:
        img = load_and_preprocess_image(image_path)
        if img is None:
            return None
        
        template_scores = {}
        max_similarity = 0.0
        
        # 各テンプレートとの類似度を計算
        for template, template_name in zip(templates, template_names):
            similarity = calculate_image_similarity(img, template)
            template_scores[f"{template_name}_match_score"] = similarity
            max_similarity = max(max_similarity, similarity)
        
        return {
            'is_matched': max_similarity > threshold,
            'template_scores': template_scores,
            'max_similarity': max_similarity
        }
    
    except Exception as e:
        print(f"画像処理エラー: {e}")
        return None
```

次のメッセージで続きを提供します。このコードの特徴は：
1. 各機能を独立した関数に分割
2. 詳細なエラーハンドリングと報告
3. 明確なドキュメント文字列
4. 単一責任の原則に従った設計
5. 再利用可能な関数構造

続けて、画像処理のメイン部分と評価機能を提供しましょうか？

# 4. テンプレートマッチング処理

本セクションでは、欠陥画像とテンプレート画像の類似度を計算し、欠陥の分類を行う処理について説明する。

## 主要な処理フロー

1. テンプレート画像の読み込み
2. 画像の前処理（グレースケール化）
3. 類似度の計算
4. マッチング結果の判定と出力

これらの処理により、欠陥画像とテンプレートの類似度に基づく分類が実現される。次のセクションでは、この分類結果の評価と最適な閾値の探索を行う。

## 関数の説明

### load_and_preprocess_image()
画像の読み込みと前処理を行う基本関数：

1. 画像の読み込み
   - テンプレート画像と入力画像で異なるパス処理
   - 読み込みエラーのハンドリング

2. 前処理
   - BGR形式からグレースケールへの変換
   - エラー時はNoneを返却

### load_templates()
テンプレート画像を一括で読み込む関数：

1. テンプレートの読み込み
   - PNG, JPG, JPEG形式に対応
   - 各テンプレートのサイズを表示

2. エラー処理
   - 読み込み失敗時の例外処理
   - 空のリストを返却

### calculate_image_similarity()
画像間の類似度を計算する関数：

1. サイズ調整
   - テンプレートが大きい場合はリサイズ
   - アスペクト比を考慮した縮小

2. 類似度計算
   - 正規化相関係数による類似度計算
   - 0~1の範囲で類似度を出力

### process_single_image()
1枚の画像に対するテンプレートマッチング処理：

1. マッチング処理
   - 全テンプレートと類似度計算
   - 最大類似度の記録

2. 結果の出力
   - マッチング判定（閾値との比較）
   - 各テンプレートのスコア
   - 最大類似度

### process_images_with_threshold()
全画像に対するテンプレートマッチング処理：

1. 処理内容
   - DataFrame内の全画像を処理
   - 進捗表示付きの一括処理

2. 出力データ
   - 予測ラベル（1/0）
   - 最大類似度
   - テンプレートごとのスコア


''' 
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

def process_images_with_threshold(df_filtered, templates, template_names, threshold):
   """
   全画像に対してテンプレートマッチングを実行します
   
   引数:
   df_filtered (pandas.DataFrame): 処理対象のデータフレーム
   templates (list): テンプレート画像のリスト
   template_names (list): テンプレート名のリスト
   threshold (float): 類似度の閾値
   
   戻り値:
   pandas.DataFrame: マッチング結果を含むデータフレーム
   """
   results = []
   for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="画像処理中"):
       try:
           # 元のデータフレームの行をコピー
           result_row = row.copy()
           
           # マッチング実行
           matching_result = process_single_image(
               row['defect_image_orig'], 
               templates, 
               template_names, 
               threshold
           )
           
           if matching_result is not None:
               # 予測結果とスコアを追加
               result_row['predicted_label'] = 1 if matching_result['is_matched'] else 0
               result_row['max_similarity'] = matching_result['max_similarity']
               
               # 各テンプレートのスコアを追加
               for score_name, score in matching_result['template_scores'].items():
                   result_row[score_name] = score
               
               results.append(result_row)
           
       except Exception as e:
           print(f"行の処理でエラー: {e}")
           continue
   
   return pd.DataFrame(results)

# テンプレートの読み込み
templates, template_names = load_templates()
'''

```python
# 5. 画像処理のメイン実装

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
```

```python
# 6. 評価指標の計算

def calculate_metrics(results_df):
    """
    精度指標を計算します
    
    引数:
    results_df (pandas.DataFrame): 予測結果を含むデータフレーム
    
    戻り値:
    dict: 各種評価指標
    """
    # 混同行列の要素を計算
    TP = sum((results_df['defect_label'] == 1) & (results_df['predicted_label'] == 1))
    FP = sum((results_df['defect_label'] == 0) & (results_df['predicted_label'] == 1))
    FN = sum((results_df['defect_label'] == 1) & (results_df['predicted_label'] == 0))
    TN = sum((results_df['defect_label'] == 0) & (results_df['predicted_label'] == 0))
    
    # 各指標の計算
    total = TP + TN + FP + FN
    detection_rate = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0
    false_detection_rate = (FP / (TN + FP)) * 100 if (TN + FP) > 0 else 0
    accuracy = ((TP + TN) / total) * 100 if total > 0 else 0
    
    return {
        'detection_rate': (detection_rate, f"{TP}/{TP+FN}"),
        'false_detection_rate': (false_detection_rate, f"{FP}/{TN+FP}"),
        'accuracy': (accuracy, f"{TP+TN}/{total}"),
        'confusion_matrix': {
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN
        }
    }
```

```python
def optimize_threshold():
    """
    最適な閾値を探索します
    1. 鋳巣検出率100%（または最大化）
    2. 鋳巣誤検出率の最小化
    
    戻り値:
    tuple: (best_threshold, results_dict)
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_threshold = None
    best_metrics = None
    best_detection_rate = -1
    best_false_rate = float('inf')
    
    results_dict = {}
    
    print("\n=== 閾値の最適化 ===")
    for threshold in tqdm(thresholds, desc="閾値探索中"):
        results_df = process_images_with_threshold(df_filtered, templates, template_names, threshold)
        metrics = calculate_metrics(results_df)
        
        detection_rate = metrics['detection_rate'][0]
        false_rate = metrics['false_detection_rate'][0]
        
        results_dict[threshold] = metrics
        
        # 最適な閾値の更新
        if detection_rate > best_detection_rate or \
           (detection_rate == best_detection_rate and false_rate < best_false_rate):
            best_detection_rate = detection_rate
            best_false_rate = false_rate
            best_threshold = threshold
            best_metrics = metrics
    
    return best_threshold, results_dict
```

```python
# メイン処理の実行
print("\n=== テンプレートマッチング実行 ===")

# テンプレートの読み込み
templates, template_names = load_templates()

if templates:
    # 閾値の最適化
    best_threshold, all_results = optimize_threshold()
    
    # 最終結果の表示
    print("\n=== 最終結果 ===")
    print(f"最適な閾値: {best_threshold:.2f}")
    best_metrics = all_results[best_threshold]
    
    print(f"\n鋳巣検出率: {best_metrics['detection_rate'][0]:.2f}% {best_metrics['detection_rate'][1]}")
    print(f"鋳巣誤検出率: {best_metrics['false_detection_rate'][0]:.2f}% {best_metrics['false_detection_rate'][1]}")
    print(f"正解率: {best_metrics['accuracy'][0]:.2f}% {best_metrics['accuracy'][1]}")
    
    # 混同行列の表示
    cm = best_metrics['confusion_matrix']
    print("\n=== 混同行列 ===")
    print(f"True Positive (TP): {cm['TP']}")
    print(f"False Positive (FP): {cm['FP']}")
    print(f"False Negative (FN): {cm['FN']}")
    print(f"True Negative (TN): {cm['TN']}")
else:
    print("テンプレート画像の読み込みに失敗しました")
```

このコードの特徴：
1. 明確な評価指標の計算と表示
2. 詳細なエラーハンドリング
3. 進捗状況の表示
4. 分母と分子の表示による透明性の確保
5. モジュール化された関数構造

改善点：
1. 各関数の責任を明確に分離
2. 評価指標の計算を独立した関数に分割
3. エラー処理の強化
4. 結果の可視化と詳細な報告
5. コードの再利用性の向上

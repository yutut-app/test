```python
def calculate_work_level_predictions(df):
    """
    ワークごとの予測と真値を計算します
    
    引数:
    df (pandas.DataFrame): 予測結果を含むデータフレーム
    
    戻り値:
    pandas.DataFrame: ワークごとの予測結果と真値
    """
    # ワークごとに集約
    work_results = df.groupby('work_id').agg({
        'defect_label': 'max',      # 1つでも欠陥があれば1
        'predicted_label': 'max'    # 1つでも欠陥と予測されれば1
    }).reset_index()
    
    # 列名の変更
    work_results.columns = ['work_id', 'true_label', 'predicted_label']
    
    return work_results

def calculate_work_level_metrics(work_results):
    """
    ワークレベルでの性能指標を計算します
    
    引数:
    work_results (pandas.DataFrame): ワークごとの予測結果
    
    戻り値:
    dict: 各種評価指標
    """
    # 混同行列の要素を計算
    work_TP = sum((work_results['true_label'] == 1) & (work_results['predicted_label'] == 1))
    work_FP = sum((work_results['true_label'] == 0) & (work_results['predicted_label'] == 1))
    work_FN = sum((work_results['true_label'] == 1) & (work_results['predicted_label'] == 0))
    work_TN = sum((work_results['true_label'] == 0) & (work_results['predicted_label'] == 0))
    
    # 各指標の計算
    total = work_TP + work_TN + work_FP + work_FN
    miss_rate = (work_FN / (work_TP + work_FN)) * 100 if (work_TP + work_FN) > 0 else 0
    over_detection_rate = (work_FP / (work_TN + work_FP)) * 100 if (work_TN + work_FP) > 0 else 0
    accuracy = ((work_TP + work_TN) / total) * 100 if total > 0 else 0
    
    return {
        'miss_rate': (miss_rate, f"{work_FN}/{work_TP+work_FN}"),
        'over_detection_rate': (over_detection_rate, f"{work_FP}/{work_TN+work_FP}"),
        'accuracy': (accuracy, f"{work_TP+work_TN}/{total}"),
        'confusion_matrix': {
            'TP': work_TP,
            'FP': work_FP,
            'FN': work_FN,
            'TN': work_TN
        }
    }

def print_work_level_metrics(metrics):
    """
    ワークレベルでの性能指標を表示します
    
    引数:
    metrics (dict): 評価指標の辞書
    """
    print("\n=== ワークレベルでの評価結果 ===")
    print(f"見逃し率: {metrics['miss_rate'][0]:.2f}% {metrics['miss_rate'][1]}")
    print(f"見過ぎ率: {metrics['over_detection_rate'][0]:.2f}% {metrics['over_detection_rate'][1]}")
    print(f"正解率: {metrics['accuracy'][0]:.2f}% {metrics['accuracy'][1]}")
    
    cm = metrics['confusion_matrix']
    print("\n=== ワークレベルでの混同行列 ===")
    print(f"True Positive (TP): {cm['TP']}")
    print(f"False Positive (FP): {cm['FP']}")
    print(f"False Negative (FN): {cm['FN']}")
    print(f"True Negative (TN): {cm['TN']}")

# メイン処理に追加
def evaluate_work_level_performance(results_df):
    """
    ワークレベルでの性能評価を実行します
    
    引数:
    results_df (pandas.DataFrame): 予測結果を含むデータフレーム
    
    戻り値:
    dict: ワークレベルでの評価指標
    """
    try:
        # ワークレベルでの予測を計算
        work_results = calculate_work_level_predictions(results_df)
        
        # 評価指標の計算
        work_metrics = calculate_work_level_metrics(work_results)
        
        # 結果の表示
        print_work_level_metrics(work_metrics)
        
        return work_metrics
    
    except Exception as e:
        print(f"ワークレベル評価でエラー: {e}")
        return None
```

```python
# メイン処理での使用例（既存のコードに追加）
if templates:
    # 閾値の最適化
    best_threshold, all_results = optimize_threshold()
    
    # 最終の予測結果を取得
    final_results_df = process_images_with_threshold(
        df_filtered, templates, template_names, best_threshold
    )
    
    # 欠陥レベルでの評価
    print("\n=== 欠陥レベルでの評価 ===")
    defect_metrics = calculate_metrics(final_results_df)
    
    print(f"鋳巣検出率: {defect_metrics['detection_rate'][0]:.2f}% {defect_metrics['detection_rate'][1]}")
    print(f"鋳巣誤検出率: {defect_metrics['false_detection_rate'][0]:.2f}% {defect_metrics['false_detection_rate'][1]}")
    print(f"正解率: {defect_metrics['accuracy'][0]:.2f}% {defect_metrics['accuracy'][1]}")
    
    # ワークレベルでの評価
    work_metrics = evaluate_work_level_performance(final_results_df)
```

特徴：
1. 欠陥レベルとワークレベルの評価を明確に分離
2. 各評価指標の計算ロジックを個別の関数に分割
3. エラーハンドリングの追加
4. 分かりやすい結果の表示
5. 分母と分子の表示による透明性の確保

改善点：
1. 関数の責任を明確に分離
2. コードの重複を削減
3. エラー処理の強化
4. 結果の可視化と詳細な報告
5. 再利用可能な関数設計

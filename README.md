# 5. 最適な閾値の探索

本セクションでは、テンプレートマッチングの判定に用いる最適な閾値を探索する処理について説明する。最適な閾値は、鋳巣の検出率を最大化しつつ、誤検出率を最小化するように決定される。

## optimize_threshold()

### 目的
- 鋳巣検出率の最大化（理想的には100%）
- 誤検出率の最小化
- 最適な判定閾値の特定

### 探索プロセス

1. 探索範囲の設定
   - 閾値：0.1から0.95まで
   - ステップ幅：0.05
   - `np.arange(0.1, 1.0, 0.05)`で生成

2. 各閾値での評価
   - テンプレートマッチングの実行
   - 精度指標の計算
   - 結果の記録と表示

3. 最適閾値の判定基準
   - 第一優先：鋳巣検出率の最大化
   - 第二優先：誤検出率の最小化
   - 同一検出率の場合は誤検出率が低い方を選択

### 出力情報

1. リアルタイム表示
   ```
   === 閾値の最適化 ===
   閾値  検出率(%)  誤検出率(%)
   ------------------------
   0.10    100.00     15.20
   0.15     98.50     12.30
   ...
   ```

2. 最終結果
   ```
   === 最適な閾値での結果 ===
   閾値: X.XX
   鋳巣検出率: XX.XX%
   鋳巣誤検出率: XX.XX%
   ```

3. 戻り値
   - best_threshold：最適な閾値
   - results_dict：全閾値での評価結果
   - best_results_df：最適閾値での詳細結果

この最適化プロセスにより、後続の評価プロセスで使用する閾値が決定される。閾値の選択は分類性能に直接影響するため、この処理は特に重要である。

def optimize_threshold(df_filtered, templates, template_names):
    """
    最適な閾値を探索します
    1. 鋳巣検出率100%（または最大化）
    2. 鋳巣誤検出率の最小化
    
    引数:
    df_filtered (pandas.DataFrame): 処理対象のデータフレーム
    templates (list): テンプレート画像のリスト
    template_names (list): テンプレート名のリスト
    
    戻り値:
    tuple: (best_threshold, results_dict, best_results_df)
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_threshold = None
    best_metrics = None
    best_detection_rate = -1
    best_false_rate = float('inf')
    best_results_df = None
    
    results_dict = {}
    
    print("\n=== 閾値の最適化 ===")
    print("閾値  検出率(%)  誤検出率(%)")
    print("-" * 35)
    
    for threshold in tqdm(thresholds, desc="閾値探索中"):
        # 現在の閾値でテンプレートマッチングを実行
        results_df = process_images_with_threshold(
            df_filtered, templates, template_names, threshold
        )
        
        # 精度指標の計算
        defect_metrics = calculate_defect_metrics(results_df)
        detection_rate = defect_metrics['detection_rate'][0]
        false_rate = defect_metrics['false_detection_rate'][0]
        
        results_dict[threshold] = defect_metrics
        
        # 結果の表示
        print(f"{threshold:.2f}  {detection_rate:8.2f}  {false_rate:8.2f}")
        
        # 最適な閾値の更新
        if detection_rate > best_detection_rate or \
           (detection_rate == best_detection_rate and false_rate < best_false_rate):
            best_detection_rate = detection_rate
            best_false_rate = false_rate
            best_threshold = threshold
            best_metrics = defect_metrics
            best_results_df = results_df
    
    print("\n=== 最適な閾値での結果 ===")
    print(f"閾値: {best_threshold:.2f}")
    print(f"鋳巣検出率: {best_detection_rate:.2f}%")
    print(f"鋳巣誤検出率: {best_false_rate:.2f}%")
    
    return best_threshold, results_dict, best_results_df

# 閾値最適化の実行
if templates:
    best_threshold, all_results, final_results_df = optimize_threshold(
        df, templates, template_names
    )
else:
    print("テンプレート画像の読み込みに失敗しました")

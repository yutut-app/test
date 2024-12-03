メイン処理の実行部分の前に、特徴量チェック用の関数を追加し、メイン処理の実行部分を修正します。

```python
def check_features(row):
    """
    特徴量が全て0（形が存在しない）かチェックする
    
    Parameters:
    row (pandas.Series): データフレームの1行
    
    Returns:
    bool: 全ての特徴量が0ならTrue
    """
    feature_columns = [
        'width', 'height', 'area', 'centroid_y', 'centroid_x',
        'perimeter', 'eccentricity', 'orientation', 'major_axis_length',
        'minor_axis_length', 'solidity', 'extent', 'aspect_ratio'
    ]
    
    # 特徴量が全て0かチェック
    return all(row[col] == 0 for col in feature_columns)

# メイン処理の実行部分を以下のように修正
print("=== データの読み込み ===")
df = pd.read_csv(defected_csv_path)
df_filtered = df[df['work_predict_label'] == 1].copy()

# 特徴量チェックを行い、予測ラベルを設定
print("\n=== 特徴量チェック ===")
df_filtered['predicted_label_features'] = df_filtered.apply(check_features, axis=1)
# 特徴量が全て0のものは予測ラベルを0に設定（形がないもの）
df_filtered['predicted_label_features'] = ~df_filtered['predicted_label_features']
print(f"形が存在しないデータ数: {sum(~df_filtered['predicted_label_features'])}")

# テンプレートの読み込み
templates, template_names = load_templates()

# 形が存在するデータのみでテンプレートマッチングを実行
df_matching = df_filtered[df_filtered['predicted_label_features'] == 1].copy()
print(f"\nテンプレートマッチング対象データ数: {len(df_matching)}")

# 閾値の最適化
print("\n=== 閾値の最適化を開始 ===")
best_threshold, all_results = optimize_threshold()

# 最終評価
print("\n=== 最適な閾値での最終評価 ===")
matching_results_df = process_images_with_threshold(df_matching, templates, template_names, best_threshold)

# 最終的な予測結果を統合
final_results_df = df_filtered.copy()
final_results_df['predicted_label'] = 0  # デフォルトで0に設定

# テンプレートマッチング結果を反映
for idx, row in matching_results_df.iterrows():
    final_results_df.loc[idx, 'predicted_label'] = row['predicted_label']
    # スコアも保存
    for template_name in template_names:
        score_col = f"{template_name}_match_score"
        if score_col in row:
            final_results_df.loc[idx, score_col] = row[score_col]

# 最終評価の計算と表示
detection_rate, false_rate = calculate_metrics(final_results_df)
print(f"最終検出率: {detection_rate * 100:.2f}%")
print(f"最終誤検出率: {false_rate * 100:.2f}%")
```

変更点の説明：
1. `check_features`関数の追加
   - 指定された特徴量が全て0かチェックする関数
   - 形が存在しないデータを判別

2. メイン処理の修正
   - データ読み込み後に特徴量チェックを実施
   - 形が存在するデータのみでテンプレートマッチングを実行
   - 最終的な予測結果を統合（特徴量が0のものは必ず予測ラベル0）

3. 結果の統合
   - 特徴量チェックとテンプレートマッチングの両方の結果を考慮
   - 全データに対する最終的な予測ラベルを設定

この修正により：
- 特徴量が全て0のデータは自動的に非欠陥（predicted_label=0）と判定
- 形が存在するデータのみにテンプレートマッチングを実行
- 処理時間の短縮（不要なマッチング処理の削減）
- より正確な欠陥判定が可能

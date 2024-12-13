# 8. 評価結果のCSV保存

本セクションでは、テンプレートマッチングと性能評価の結果をCSVファイルとして保存する処理について説明する。

## 実装の詳細

### prepare_results_for_save()
保存用のデータを準備する関数：

1. 結果データフレームの作成
   - 元データのコピー作成
   - テンプレートごとのマッチングスコア追加
   - 予測ラベルと最大類似度の追加

2. 性能評価データフレームの作成
   ```
   metrics_data = {
       'Metric': 評価指標名
       'Value': 数値結果
       'Count': カウント情報
   }
   ```

3. 格納される評価指標
   - 欠陥レベル：
     - 鋳巣検出率
     - 鋳巣誤検出率
     - 正解率
   - ワークレベル：
     - 見逃し率
     - 見過ぎ率
     - 正解率

### save_results_to_csv()
結果をCSVファイルとして保存する関数：

1. ファイル名の生成
   - 基本名：template_matching_results
   - タイムスタンプ付加：YYYYMMDD_HHMMSS
   - 2種類のファイル生成：
     - 結果データ：{base_filename}_{timestamp}.csv
     - 性能評価：{base_filename}_metrics_{timestamp}.csv

2. 保存内容
   - 結果データ：
     - 元の特徴量データ
     - テンプレートマッチングスコア
     - 予測結果
     - 最大類似度

   - 性能評価データ：
     - 評価指標名
     - 計算結果
     - カウント情報

3. 確認出力
   - 保存したファイル名
   - 結果データの列構成
   - 性能評価データの内容

def prepare_results_for_save(df_original, final_results_df, template_names, defect_metrics, work_metrics):
   """
   保存用のデータフレームを準備します
   
   引数:
   df_original (pandas.DataFrame): 元のデータフレーム
   final_results_df (pandas.DataFrame): テンプレートマッチング結果
   template_names (list): テンプレート名のリスト
   defect_metrics (dict): 欠陥レベルの性能評価結果
   work_metrics (dict): ワークレベルの性能評価結果
   
   戻り値:
   tuple: (結果のデータフレーム, 性能評価のデータフレーム)
   """
   try:
       # 結果データフレームの準備
       result_df = df_original.copy()
       
       # テンプレートマッチングの結果を追加
       for template_name in template_names:
           score_column = f"{template_name}_match_score"
           if score_column in final_results_df.columns:
               result_df[score_column] = final_results_df[score_column]
       
       # 予測ラベルと最大類似度を追加
       if 'predicted_label' in final_results_df.columns:
           result_df['predicted_label'] = final_results_df['predicted_label']
       if 'max_similarity' in final_results_df.columns:
           result_df['max_similarity'] = final_results_df['max_similarity']
       
       # 性能評価結果の準備
       metrics_data = {
           'Metric': [],
           'Value': [],
           'Count': []
       }
       
       # 欠陥レベルの性能評価
       metrics_data['Metric'].append('鋳巣検出率')
       metrics_data['Value'].append(f"{defect_metrics['detection_rate'][0]:.2f}%")
       metrics_data['Count'].append(defect_metrics['detection_rate'][1])
       
       metrics_data['Metric'].append('鋳巣誤検出率')
       metrics_data['Value'].append(f"{defect_metrics['false_detection_rate'][0]:.2f}%")
       metrics_data['Count'].append(defect_metrics['false_detection_rate'][1])
       
       metrics_data['Metric'].append('欠陥レベル正解率')
       metrics_data['Value'].append(f"{defect_metrics['accuracy'][0]:.2f}%")
       metrics_data['Count'].append(defect_metrics['accuracy'][1])
       
       # ワークレベルの性能評価
       metrics_data['Metric'].append('見逃し率')
       metrics_data['Value'].append(f"{work_metrics['miss_rate'][0]:.2f}%")
       metrics_data['Count'].append(work_metrics['miss_rate'][1])
       
       metrics_data['Metric'].append('見過ぎ率')
       metrics_data['Value'].append(f"{work_metrics['over_detection_rate'][0]:.2f}%")
       metrics_data['Count'].append(work_metrics['over_detection_rate'][1])
       
       metrics_data['Metric'].append('ワークレベル正解率')
       metrics_data['Value'].append(f"{work_metrics['accuracy'][0]:.2f}%")
       metrics_data['Count'].append(work_metrics['accuracy'][1])
       
       metrics_df = pd.DataFrame(metrics_data)
       
       return result_df, metrics_df
       
   except Exception as e:
       print(f"結果準備でエラー: {e}")
       return None, None

def save_results_to_csv(result_df, metrics_df, base_filename='template_matching_results'):
   """
   結果をCSVファイルとして保存します
   
   引数:
   result_df (pandas.DataFrame): 結果のデータフレーム
   metrics_df (pandas.DataFrame): 性能評価のデータフレーム
   base_filename (str): 保存するファイル名の基本部分
   """
   try:
       # タイムスタンプの生成
       timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
       
       # 結果の保存
       results_filename = f"{base_filename}_{timestamp}.csv"
       result_df.to_csv(results_filename, index=False)
       print(f"結果を保存しました: {results_filename}")
       
       # 性能評価の保存
       metrics_filename = f"{base_filename}_metrics_{timestamp}.csv"
       metrics_df.to_csv(metrics_filename, index=False)
       print(f"性能評価を保存しました: {metrics_filename}")
       
   except Exception as e:
       print(f"保存でエラー: {e}")

# メイン処理での使用
if templates and final_results_df is not None:
   # 結果の準備
   result_df, metrics_df = prepare_results_for_save(
       df,
       final_results_df,
       template_names,
       defect_metrics,
       work_metrics
   )
   
   if result_df is not None and metrics_df is not None:
       # 保存の実行
       save_results_to_csv(result_df, metrics_df)
       
       # 保存内容の確認
       print("\n=== 保存したデータの確認 ===")
       print("\n結果データフレームの列:")
       print(result_df.columns.tolist())
       
       print("\n性能評価データ:")
       print(metrics_df)
   else:
       print("結果の準備に失敗しました")
else:
   print("テンプレートマッチングの結果がありません")

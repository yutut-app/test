# 6. 欠陥ごとのモデル評価

本セクションでは、欠陥（鋳巣）単位での分類性能を評価する。各欠陥に対する検出精度を詳細に分析し、モデルの性能を定量的に評価する。

## 評価指標の解釈

1. 鋳巣検出率
   - 実際の鋳巣をどれだけ検出できたか
   - 高いほど見逃しが少ない

2. 鋳巣誤検出率
   - 非鋳巣をどれだけ誤って鋳巣と判定したか
   - 低いほど過剰検出が少ない

3. 正解率
   - 全判定のうち、正しい判定の割合
   - 総合的な性能指標

これらの評価により、欠陥検出システムの性能を定量的に把握し、必要に応じたパラメータの調整や改善点の特定が可能となる。

## 実装の詳細

### calculate_defect_metrics()
欠陥レベルでの各種性能指標を計算する関数：

1. 混同行列の計算
   - True Positive (TP)：正しく検出された鋳巣
   - False Positive (FP)：誤って検出された鋳巣
   - False Negative (FN)：見逃された鋳巣
   - True Negative (TN)：正しく非鋳巣と判定

2. 評価指標の算出
   - 鋳巣検出率 = TP / (TP + FN) × 100
   - 鋳巣誤検出率 = FP / (TN + FP) × 100
   - 正解率 = (TP + TN) / 総数 × 100

3. エラー処理
   - 分母が0となる場合の処理
   - 例外発生時のエラーハンドリング

### print_defect_metrics()
評価結果を可読性の高い形式で表示する関数：

1. 基本指標の表示
   ```
   === 欠陥レベルでの評価結果 ===
   鋳巣検出率: XX.XX% (検出数/総数)
   鋳巣誤検出率: XX.XX% (誤検出数/非欠陥総数)
   正解率: XX.XX% (正解数/総数)
   ```

2. 混同行列の表示
   ```
   === 混同行列 ===
   True Positive (TP): XX
   False Positive (FP): XX
   False Negative (FN): XX
   True Negative (TN): XX
   ```

### evaluate_defect_level_performance()
欠陥レベルでの性能評価を実行する関数：

1. 処理フロー
   - 評価指標の計算
   - 結果の表示
   - 指標のデータ保持

2. エラー処理
   - 計算失敗時の処理
   - 異常値のハンドリング


def calculate_defect_metrics(results_df):
    """
    欠陥レベルでの性能指標を計算します
    
    引数:
    results_df (pandas.DataFrame): 予測結果を含むデータフレーム
    
    戻り値:
    dict: 各種評価指標と混同行列
    """
    try:
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
        
    except Exception as e:
        print(f"欠陥レベルの評価でエラー: {e}")
        return None

def print_defect_metrics(metrics):
    """
    欠陥レベルでの評価結果を表示します
    
    引数:
    metrics (dict): 評価指標の辞書
    """
    print("\n=== 欠陥レベルでの評価結果 ===")
    print(f"鋳巣検出率: {metrics['detection_rate'][0]:.2f}% {metrics['detection_rate'][1]}")
    print(f"鋳巣誤検出率: {metrics['false_detection_rate'][0]:.2f}% {metrics['false_detection_rate'][1]}")
    print(f"正解率: {metrics['accuracy'][0]:.2f}% {metrics['accuracy'][1]}")
    
    cm = metrics['confusion_matrix']
    print("\n=== 混同行列 ===")
    print(f"True Positive (TP): {cm['TP']}")
    print(f"False Positive (FP): {cm['FP']}")
    print(f"False Negative (FN): {cm['FN']}")
    print(f"True Negative (TN): {cm['TN']}")

def evaluate_defect_level_performance(results_df):
    """
    欠陥レベルでの性能評価を実行します
    
    引数:
    results_df (pandas.DataFrame): 予測結果を含むデータフレーム
    
    戻り値:
    dict: 欠陥レベルでの評価指標
    """
    try:
        # 評価指標の計算
        defect_metrics = calculate_defect_metrics(results_df)
        
        if defect_metrics is not None:
            # 結果の表示
            print_defect_metrics(defect_metrics)
            return defect_metrics
        else:
            print("評価指標の計算に失敗しました")
            return None
            
    except Exception as e:
        print(f"欠陥レベル評価でエラー: {e}")
        return None

# 欠陥レベルの評価実行
if final_results_df is not None:
    defect_metrics = evaluate_defect_level_performance(final_results_df)
else:
    print("予測結果がありません")

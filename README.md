# 7. ワークごとのモデル評価

本セクションでは、ワーク単位での分類性能を評価する。個々の欠陥ではなく、ワーク全体としての良品/不良品の判定精度を分析する。

## 評価指標の解釈

1. 見逃し率
   - 実際の不良品を良品と判定した割合
   - 品質保証の観点で特に重要

2. 見過ぎ率
   - 良品を不良品と誤判定した割合
   - 生産効率に影響

3. 正解率
   - 全体的な判定精度
   - システムの総合的な性能指標

これらの評価により、実際の生産ラインでの運用を想定した性能評価が可能となる。

## 実装の詳細

### calculate_work_level_predictions()
ワークごとの予測結果を集約する関数：

1. 集約方法
   - work_idでグループ化
   - 欠陥ラベル：最大値を採用（1つでも欠陥があれば1）
   - 予測ラベル：最大値を採用（1つでも欠陥と予測されれば1）

2. データ構造
   ```python
   work_results:
   - work_id: ワークの識別子
   - true_label: 実際の良品/不良品ラベル
   - predicted_label: 予測された良品/不良品ラベル
   ```

### calculate_work_metrics()
ワークレベルでの性能指標を計算する関数：

1. 混同行列の要素
   - TP：正しく不良品と判定されたワーク
   - FP：誤って不良品と判定されたワーク
   - FN：見逃された不良品ワーク
   - TN：正しく良品と判定されたワーク

2. 評価指標
   - 見逃し率（FNR）= FN / (TP + FN) × 100
   - 見過ぎ率（FPR）= FP / (TN + FP) × 100
   - 正解率（ACC）= (TP + TN) / 総数 × 100

### print_work_metrics()
ワークレベルでの評価結果を表示する関数：

1. 基本指標の表示
   ```
   === ワークレベルでの評価結果 ===
   見逃し率: XX.XX% (見逃し数/不良品総数)
   見過ぎ率: XX.XX% (過検出数/良品総数)
   正解率: XX.XX% (正解数/総数)
   ```

2. 混同行列の詳細表示
   ```
   === ワークレベルでの混同行列 ===
   True Positive (TP): XX
   False Positive (FP): XX
   False Negative (FN): XX
   True Negative (TN): XX
   ```

### evaluate_work_level_performance()
ワークレベルでの総合的な性能評価を実行する関数：

1. 処理フロー
   - ワーク単位での予測集約
   - 評価指標の計算
   - 結果の表示と保存

2. エラー処理
   - 予測集約失敗時の処理
   - 評価指標計算失敗時の処理

def calculate_work_level_predictions(df):
    """
    ワークごとの予測と真値を計算します
    
    引数:
    df (pandas.DataFrame): 予測結果を含むデータフレーム
    
    戻り値:
    pandas.DataFrame: ワークごとの予測結果と真値
    """
    try:
        # ワークごとに集約
        work_results = df.groupby('work_id').agg({
            'defect_label': 'max',      # 1つでも欠陥があれば1
            'predicted_label': 'max'    # 1つでも欠陥と予測されれば1
        }).reset_index()
        
        # 列名の変更
        work_results.columns = ['work_id', 'true_label', 'predicted_label']
        
        return work_results
        
    except Exception as e:
        print(f"ワークレベルの予測集約でエラー: {e}")
        return None

def calculate_work_metrics(work_results):
    """
    ワークレベルでの性能指標を計算します
    
    引数:
    work_results (pandas.DataFrame): ワークごとの予測結果
    
    戻り値:
    dict: 各種評価指標と混同行列
    """
    try:
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
                'TP': work_TP, 'FP': work_FP, 'FN': work_FN, 'TN': work_TN
            }
        }
        
    except Exception as e:
        print(f"ワークレベルの評価指標計算でエラー: {e}")
        return None

def print_work_metrics(metrics):
    """
    ワークレベルでの評価結果を表示します
    
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
        
        if work_results is not None:
            # 評価指標の計算
            work_metrics = calculate_work_metrics(work_results)
            
            if work_metrics is not None:
                # 結果の表示
                print_work_metrics(work_metrics)
                return work_metrics
            else:
                print("ワークレベルの評価指標計算に失敗しました")
                return None
        else:
            print("ワークレベルの予測集約に失敗しました")
            return None
            
    except Exception as e:
        print(f"ワークレベル評価でエラー: {e}")
        return None

# ワークレベルの評価実行
if final_results_df is not None:
    work_metrics = evaluate_work_level_performance(final_results_df)
else:
    print("予測結果がありません")

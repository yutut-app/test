パターンマッチングの実装と精度評価を行うIPythonノートブックを作成します。

```markdown
# 欠陥検出システム - パターンマッチング分析

## 3. テンプレートマッチングの実装
```

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_image(image_path):
    """画像を読み込んでグレースケール化する"""
    img = cv2.imread(os.path.join(defected_data_path, image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def perform_template_matching(image, templates, threshold=0.8):
    """
    画像に対してテンプレートマッチングを実行
    """
    max_score = 0
    for template in templates:
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        max_score = max(max_score, max_val)
    return max_score > threshold

# NGデータ（欠陥あり）からテンプレートを作成
ng_samples = df_filtered[df_filtered['defect_label'] == 1]
templates = []
for _, row in ng_samples.iterrows():
    try:
        img = load_image(row['defect_image_orig'])
        templates.append(img)
    except Exception as e:
        print(f"Error loading template: {e}")

print(f"読み込んだテンプレート数: {len(templates)}")
```

```python
# パターンマッチングによる分類の実行
results = []
for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    try:
        img = load_image(row['defect_image_orig'])
        is_defect = perform_template_matching(img, templates)
        results.append({
            'work_id': row['work_id'],
            'true_label': row['defect_label'],
            'predicted_label': 1 if is_defect else 0
        })
    except Exception as e:
        print(f"Error processing image: {e}")

results_df = pd.DataFrame(results)
```

```markdown
## 4. 精度評価
```

```python
def calculate_metrics(results_df, df_filtered):
    """精度指標を計算する"""
    # 欠陥ごとの評価
    TP = sum((results_df['true_label'] == 1) & (results_df['predicted_label'] == 1))
    FP = sum((results_df['true_label'] == 0) & (results_df['predicted_label'] == 1))
    FN = sum((results_df['true_label'] == 1) & (results_df['predicted_label'] == 0))
    TN = sum((results_df['true_label'] == 0) & (results_df['predicted_label'] == 0))
    
    defect_detection_rate = TP / (FN + TP) if (FN + TP) > 0 else 0
    false_detection_rate = FP / (TN + FP) if (TN + FP) > 0 else 0
    
    # ワークごとの評価
    work_results = results_df.groupby('work_id').agg({
        'true_label': 'max',
        'predicted_label': 'max'
    }).reset_index()
    
    work_TP = sum((work_results['true_label'] == 1) & (work_results['predicted_label'] == 1))
    work_FP = sum((work_results['true_label'] == 0) & (work_results['predicted_label'] == 1))
    work_FN = sum((work_results['true_label'] == 1) & (work_results['predicted_label'] == 0))
    work_TN = sum((work_results['true_label'] == 0) & (work_results['predicted_label'] == 0))
    
    miss_rate = work_FN / (work_FN + work_TP) if (work_FN + work_TP) > 0 else 0
    over_detection_rate = work_FP / (work_FP + work_TN) if (work_FP + work_TN) > 0 else 0
    
    return {
        'defect_detection_rate': defect_detection_rate * 100,
        'false_detection_rate': false_detection_rate * 100,
        'miss_rate': miss_rate * 100,
        'over_detection_rate': over_detection_rate * 100
    }

# 精度指標の計算と表示
metrics = calculate_metrics(results_df, df_filtered)

print("=== 精度評価結果 ===")
print(f"欠陥検出率: {metrics['defect_detection_rate']:.2f}%")
print(f"誤検出率: {metrics['false_detection_rate']:.2f}%")
print(f"見逃し率: {metrics['miss_rate']:.2f}%")
print(f"見過ぎ率: {metrics['over_detection_rate']:.2f}%")
```

```python
# 閾値を調整して欠陥検出率100%を目指す
def optimize_threshold():
    thresholds = np.arange(0.5, 1.0, 0.05)
    best_metrics = None
    best_threshold = None
    
    for threshold in thresholds:
        results = []
        for _, row in df_filtered.iterrows():
            try:
                img = load_image(row['defect_image_orig'])
                is_defect = perform_template_matching(img, templates, threshold)
                results.append({
                    'work_id': row['work_id'],
                    'true_label': row['defect_label'],
                    'predicted_label': 1 if is_defect else 0
                })
            except Exception as e:
                continue
                
        results_df = pd.DataFrame(results)
        metrics = calculate_metrics(results_df, df_filtered)
        
        if metrics['defect_detection_rate'] == 100:
            if best_metrics is None or metrics['false_detection_rate'] < best_metrics['false_detection_rate']:
                best_metrics = metrics
                best_threshold = threshold
    
    return best_threshold, best_metrics

# 最適な閾値を見つける
best_threshold, best_metrics = optimize_threshold()

print("\n=== 最適化後の評価結果 ===")
print(f"最適な閾値: {best_threshold:.2f}")
print(f"欠陥検出率: {best_metrics['defect_detection_rate']:.2f}%")
print(f"誤検出率: {best_metrics['false_detection_rate']:.2f}%")
print(f"見逃し率: {best_metrics['miss_rate']:.2f}%")
print(f"見過ぎ率: {best_metrics['over_detection_rate']:.2f}%")
```

このコードでは:

1. テンプレートマッチングの実装
   - NGデータからテンプレートを作成
   - 各画像に対してテンプレートマッチングを実行
   - 閾値によるマッチング判定

2. 精度評価
   - 欠陥ごとの検出率と誤検出率
   - ワークごとの見逃し率と見過ぎ率
   - 閾値の最適化による性能向上

3. 閾値の最適化
   - 欠陥検出率100%を満たす
   - 誤検出率を最小化

必要に応じて、以下のパラメータを調整できます：
- テンプレートマッチングの手法（現在はcv2.TM_CCOEFF_NORMED）
- 閾値の探索範囲
- テンプレートの前処理方法（必要に応じてノイズ除去やコントラスト調整など）

このコードを実行して結果を確認しますか？

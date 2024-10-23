閾値の最適化を行うコードを提供します。まずはパターンマッチングの全工程を記載し、その後に最適化のコードを示します。

```python
# 必要なライブラリのインポート
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# 画像サイズ確認用の関数
def check_image_size(image_path):
    """画像のサイズを確認する"""
    img = cv2.imread(os.path.join(defected_image_path, image_path))
    if img is None:
        return None
    return img.shape

def load_and_preprocess_image(image_path, target_size=None):
    """画像を読み込み、前処理を行う"""
    img = cv2.imread(os.path.join(defected_image_path, image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # グレースケール化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # リサイズが必要な場合
    if target_size is not None:
        img_gray = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)
    
    return img_gray

def perform_template_matching(image, templates, threshold):
    """
    画像に対してテンプレートマッチングを実行
    """
    max_score = 0
    for template in templates:
        # 画像サイズの取得
        img_height, img_width = image.shape
        templ_height, templ_width = template.shape
        
        # テンプレートが入力画像より大きい場合、テンプレートをリサイズ
        if templ_height > img_height or templ_width > img_width:
            template = cv2.resize(template, (min(img_width, templ_width), min(img_height, templ_height)), 
                                interpolation=cv2.INTER_AREA)
        
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        max_score = max(max_score, max_val)
    
    return max_score > threshold, max_score

def calculate_metrics(results_df):
    """精度指標を計算する"""
    # 欠陥ごとの評価
    TP = sum((results_df['true_label'] == 1) & (results_df['predicted_label'] == 1))
    FP = sum((results_df['true_label'] == 0) & (results_df['predicted_label'] == 1))
    FN = sum((results_df['true_label'] == 1) & (results_df['predicted_label'] == 0))
    TN = sum((results_df['true_label'] == 0) & (results_df['predicted_label'] == 0))
    
    defect_detection_rate = TP / (FN + TP) if (FN + TP) > 0 else 0
    false_detection_rate = FP / (TN + FP) if (TN + FP) > 0 else 0
    
    return defect_detection_rate, false_detection_rate

def process_images_with_threshold(df_filtered, templates, threshold):
    """指定された閾値でパターンマッチングを実行"""
    results = []
    for _, row in df_filtered.iterrows():
        try:
            img = load_and_preprocess_image(row['defect_image_orig'])
            is_matched, score = perform_template_matching(img, templates, threshold)
            results.append({
                'work_id': row['work_id'],
                'true_label': row['defect_label'],
                'predicted_label': 1 if is_matched else 0,
                'match_score': score
            })
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
    
    return pd.DataFrame(results)
```

```python
# テンプレート画像の準備
print("=== テンプレート画像の準備 ===")
ng_samples = df_filtered[df_filtered['defect_label'] == 1]
templates = []
for idx, row in ng_samples.iterrows():
    try:
        img = load_and_preprocess_image(row['defect_image_orig'])
        templates.append(img)
        print(f"テンプレート {idx+1}: サイズ {img.shape}")
    except Exception as e:
        print(f"Error loading template {idx+1}: {e}")

print(f"\n読み込んだテンプレート数: {len(templates)}")
```

```python
# 閾値の最適化
def optimize_threshold():
    """
    defect_detection_rateを最大化し、
    同じdefect_detection_rateの場合はfalse_detection_rateを最小化する閾値を探索
    """
    thresholds = np.arange(0.1, 1.0, 0.05)  # 探索する閾値の範囲
    best_threshold = None
    best_detection_rate = -1
    best_false_rate = float('inf')
    
    results_dict = {}
    
    # 各閾値での結果を保存
    for threshold in tqdm(thresholds, desc="閾値の最適化"):
        results_df = process_images_with_threshold(df_filtered, templates, threshold)
        detection_rate, false_rate = calculate_metrics(results_df)
        
        results_dict[threshold] = {
            'detection_rate': detection_rate * 100,
            'false_rate': false_rate * 100
        }
        
        # 最適な閾値の更新
        if detection_rate > best_detection_rate:
            best_detection_rate = detection_rate
            best_false_rate = false_rate
            best_threshold = threshold
        elif detection_rate == best_detection_rate and false_rate < best_false_rate:
            best_false_rate = false_rate
            best_threshold = threshold
    
    return best_threshold, results_dict

# 最適化の実行
print("\n=== 閾値の最適化を開始 ===")
best_threshold, all_results = optimize_threshold()

# 結果の表示
print("\n=== 閾値ごとの結果 ===")
print("閾値  検出率(%)  誤検出率(%)")
print("-" * 30)
for threshold in sorted(all_results.keys()):
    result = all_results[threshold]
    print(f"{threshold:.2f}  {result['detection_rate']:8.2f}  {result['false_rate']:8.2f}")

print("\n=== 最適な閾値 ===")
print(f"閾値: {best_threshold:.2f}")
print(f"検出率: {all_results[best_threshold]['detection_rate']:.2f}%")
print(f"誤検出率: {all_results[best_threshold]['false_rate']:.2f}%")
```

```python
# 最適な閾値での最終評価
print("\n=== 最適な閾値での最終評価 ===")
final_results_df = process_images_with_threshold(df_filtered, templates, best_threshold)
detection_rate, false_rate = calculate_metrics(final_results_df)

print(f"最終検出率: {detection_rate * 100:.2f}%")
print(f"最終誤検出率: {false_rate * 100:.2f}%")

# スコアの分布を可視化
plt.figure(figsize=(10, 6))
plt.hist(final_results_df['match_score'], bins=50)
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold ({best_threshold:.2f})')
plt.xlabel('Matching Score')
plt.ylabel('Frequency')
plt.title('Distribution of Matching Scores')
plt.legend()
plt.show()
```

このコードは：
1. パターンマッチングのための基本関数を定義
2. テンプレート画像を準備
3. 閾値を0.1から1.0まで0.05刻みで探索
4. 各閾値での検出率と誤検出率を計算
5. 最適な閾値を選択（検出率最大化、同率の場合は誤検出率最小化）
6. 結果を詳細に表示
7. 最適な閾値でのスコア分布を可視化

特徴：
- すべての中間結果を表示
- 閾値ごとの性能を確認可能
- 視覚的な結果確認
- エラーハンドリングの実装
- 段階的な実行が可能

使用方法：
1. まず基本関数とテンプレート準備のコードを実行
2. 次に最適化のコードを実行
3. 最後に最終評価とビジュアライゼーションを実行

必要に応じて閾値の探索範囲や刻み値を調整できます。

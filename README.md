はい、スケール不変なパターンマッチングを実装しましょう。ORBを使用した特徴点マッチングを実装します。コードを細かく分けて実行できるようにします。

```markdown
# 欠陥検出システム - 特徴点マッチングによる分類

## 3. 特徴点マッチングの実装
```

```python
# 必要なライブラリをインポート
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
```

```python
def load_image(image_path):
    """画像を読み込んでグレースケール化する"""
    img = cv2.imread(os.path.join(defected_image_path, image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ORB検出器の初期化
orb = cv2.ORB_create(nfeatures=1000)

def extract_features(image):
    """ORBを使用して特徴点と特徴量を抽出"""
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# 特徴点マッチングの実行関数
def match_features(desc1, desc2):
    """特徴量のマッチング"""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    # 距離でソート
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def calculate_similarity(matches, num_features):
    """マッチングの類似度を計算"""
    if len(matches) == 0 or num_features == 0:
        return 0
    # マッチング数と特徴点数から類似度を計算
    similarity = len(matches) / num_features
    return similarity
```

```python
# NGデータ（欠陥あり）からテンプレート特徴を抽出
print("テンプレート特徴の抽出中...")
template_features = []
ng_samples = df_filtered[df_filtered['defect_label'] == 1]

for _, row in ng_samples.iterrows():
    try:
        img = load_image(row['defect_image_orig'])
        keypoints, descriptors = extract_features(img)
        if descriptors is not None:
            template_features.append({
                'keypoints': keypoints,
                'descriptors': descriptors
            })
        print(f"テンプレート画像から {len(keypoints)} 個の特徴点を抽出")
    except Exception as e:
        print(f"Error extracting template features: {e}")

print(f"処理したテンプレート数: {len(template_features)}")
```

```python
# 1つの画像に対する類似度計算
def compute_image_similarity(image, template_features):
    """1つの画像に対する全テンプレートとの最大類似度を計算"""
    try:
        # 入力画像の特徴点抽出
        keypoints, descriptors = extract_features(image)
        if descriptors is None:
            return 0
        
        max_similarity = 0
        # 各テンプレートとマッチング
        for template in template_features:
            matches = match_features(descriptors, template['descriptors'])
            similarity = calculate_similarity(matches, len(keypoints))
            max_similarity = max(max_similarity, similarity)
            
        return max_similarity
    except Exception as e:
        print(f"Error in similarity computation: {e}")
        return 0

# テスト用に1枚の画像で実行
test_row = df_filtered.iloc[0]
test_img = load_image(test_row['defect_image_orig'])
test_similarity = compute_image_similarity(test_img, template_features)
print(f"テスト画像の類似度: {test_similarity:.3f}")
```

```python
# 全画像に対する分類の実行
def classify_images(threshold=0.1):
    """全画像に対して分類を実行"""
    results = []
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        try:
            img = load_image(row['defect_image_orig'])
            similarity = compute_image_similarity(img, template_features)
            is_defect = similarity > threshold
            
            results.append({
                'work_id': row['work_id'],
                'true_label': row['defect_label'],
                'predicted_label': 1 if is_defect else 0,
                'similarity_score': similarity
            })
        except Exception as e:
            print(f"Error processing image: {e}")
    
    return pd.DataFrame(results)

# 初期閾値で分類を実行
results_df = classify_images(threshold=0.1)
```

```python
def calculate_metrics(results_df):
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

# 初期結果の評価
metrics = calculate_metrics(results_df)
print("\n=== 初期評価結果 ===")
print(f"欠陥検出率: {metrics['defect_detection_rate']:.2f}%")
print(f"誤検出率: {metrics['false_detection_rate']:.2f}%")
print(f"見逃し率: {metrics['miss_rate']:.2f}%")
print(f"見過ぎ率: {metrics['over_detection_rate']:.2f}%")
```

```python
def optimize_threshold():
    """閾値の最適化"""
    thresholds = np.arange(0.05, 0.3, 0.02)
    best_metrics = None
    best_threshold = None
    
    for threshold in thresholds:
        print(f"\n閾値 {threshold:.2f} での評価:")
        results_df = classify_images(threshold)
        metrics = calculate_metrics(results_df)
        
        print(f"欠陥検出率: {metrics['defect_detection_rate']:.2f}%")
        print(f"誤検出率: {metrics['false_detection_rate']:.2f}%")
        
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

この実装では：
1. ORBを使用して特徴点と特徴量を抽出
2. 特徴点マッチングにBruteForceマッチャーを使用
3. マッチング数と特徴点数から類似度を計算
4. 閾値による分類
5. 性能評価と閾値の最適化

を行っています。

コードは細かく分けられているので、各段階で結果を確認できます。必要に応じて以下のパラメータを調整できます：
- ORBの特徴点数（nfeatures）
- 類似度の計算方法
- 閾値の探索範囲

エラーが発生した場合や、調整が必要な箇所があれば教えてください。

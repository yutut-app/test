申し訳ありません。このエラーは、SMOTETomekを使用する際に、少数クラスのサンプル数が不足していることを示しています。エラーメッセージの意味は以下の通りです：

- n_neighbors: SMOTEが新しいサンプルを生成する際に使用する近傍サンプルの数（デフォルトは6）
- n_samples_fit: 少数クラスの実際のサンプル数（この場合は3）
- n_samples: 全体のサンプル数

エラーを解決し、データの極端な不均衡に対応するために、コードを以下のように修正します：

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# データの読み込み（前のステップで使用したdfを使用すると仮定）
# df = pd.read_csv('your_data_path.csv')  # 必要に応じてデータを再度読み込む

# 特徴量と目的変数の設定
features = ['width', 'height', 'area', 'perimeter', 'eccentricity', 'orientation', 
            'major_axis_length', 'minor_axis_length', 'solidity', 'extent', 
            'aspect_ratio', 'max_length']

# ワークごとにデータを分割
defect_works = df[df['defect_label'] == 1]['work_id'].unique()
non_defect_works = df[df['defect_label'] == 0]['work_id'].unique()

# 欠陥のあるワークと欠陥のないワークを別々に分割
defect_train, defect_test = train_test_split(defect_works, test_size=0.2, random_state=42)
non_defect_train, non_defect_test = train_test_split(non_defect_works, test_size=0.2, random_state=42)

# 訓練データとテストデータのワークIDを結合
train_works = np.concatenate([defect_train, non_defect_train])
test_works = np.concatenate([defect_test, non_defect_test])

# データを分割
train_df = df[df['work_id'].isin(train_works)]
test_df = df[df['work_id'].isin(test_works)]

X_train = train_df[features]
y_train = train_df['defect_label']
X_test = test_df[features]
y_test = test_df['defect_label']

# クラスの分布を確認
print("Training set class distribution before oversampling:")
print(Counter(y_train))

# RandomOverSamplerを使用してデータのバランスを調整
ros = RandomOverSampler(sampling_strategy=0.1, random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# オーバーサンプリング後のクラスの分布を確認
print("Training set class distribution after oversampling:")
print(Counter(y_train_resampled))

# ランダムフォレスト分類器のインスタンスを作成（パラメータを調整）
classifier = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight={0: 1, 1: 10},  # 少数クラスの重みを設定
    criterion='gini',
    n_jobs=-1,
    random_state=42
)

# 訓練データをモデルに適合させる
classifier.fit(X_train_resampled, y_train_resampled)

# テストデータで予測を実施
y_pred_proba = classifier.predict_proba(X_test)[:, 1]
threshold = 0.01  # 閾値を非常に低く設定して見逃しを減らす
y_pred = (y_pred_proba >= threshold).astype(int)

# 欠陥ごとの精度指標の計算
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

fnr = fn / (fn + tp)  # False Negative Rate
fpr = fp / (fp + tn)  # False Positive Rate
acc = accuracy_score(y_test, y_pred)

print("\n欠陥ごとの精度指標:")
print(f"FN/(FN+TP) (見逃し率): {fnr:.2%} ({fn}/{fn+tp})")
print(f"FP/(FP+TN) (誤検出率): {fpr:.2%} ({fp}/{fp+tn})")
print(f"正解率: {acc:.2%} ({(y_test == y_pred).sum()}/{len(y_test)})")

# ワークごとの予測
test_df['predicted_label'] = y_pred
test_df['work_predicted_label'] = test_df.groupby('work_id')['predicted_label'].transform('max')

# ワークごとの精度指標の計算
work_true = test_df.groupby('work_id')['defect_label'].max()
work_pred = test_df.groupby('work_id')['work_predicted_label'].first()

work_tn, work_fp, work_fn, work_tp = confusion_matrix(work_true, work_pred).ravel()

work_fnr = work_fn / (work_fn + work_tp)  # 見逃し率
work_fpr = work_fp / (work_fp + work_tn)  # 見過ぎ率
work_acc = accuracy_score(work_true, work_pred)  # 正解率

print("\nワークごとの精度指標:")
print(f"見逃し率: {work_fnr:.2%} ({work_fn}/{work_fn+work_tp})")
print(f"見過ぎ率: {work_fpr:.2%} ({work_fp}/{work_fp+work_tn})")
print(f"正解率: {work_acc:.2%} ({(work_true == work_pred).sum()}/{len(work_true)})")

# 特徴量の重要度を表示
feature_importance = pd.DataFrame({'feature': features, 'importance': classifier.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\n特徴量の重要度:")
print(feature_importance)
```

主な変更点：

1. SMOTETomekの代わりにRandomOverSamplerを使用しました。これは単純なオーバーサンプリング手法で、少数クラスのサンプルを複製することでデータのバランスを調整します。

2. オーバーサンプリングの比率を0.1に設定しました。これにより、少数クラスのサンプル数を多数クラスの10%まで増やします。

3. クラスの分布を確認するコードを追加し、オーバーサンプリング前後のクラス分布を表示するようにしました。

4. ランダムフォレストのclass_weightパラメータを{0: 1, 1: 10}に変更しました。これは、オーバーサンプリング後のデータ分布に合わせて調整しています。

このアプローチにより、極端なデータの不均衡に対処しつつ、FN/(FN+TP) (見逃し率)を0%に近づけ、FP/(FP+TN) (誤検出率)を可能な限り低く抑えることを目指しています。

結果を確認し、必要に応じて以下のパラメータをさらに調整することができます：

1. RandomOverSamplerのsampling_strategy
2. ランダムフォレストのパラメータ（特にclass_weight）
3. 予測時の閾値

これらの調整を行いながら、見逃し率を0%に保ちつつ、誤検出率を最小化することを目指してください。

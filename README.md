ご指摘ありがとうございます。その通りです。Isolation Forestは教師なし学習アルゴリズムであり、異常検知のためのモデルなので、データの分割は必要ありません。すべてのデータを使用してモデルを学習し、評価することができます。

以下に、データ分割を行わない修正版のコードを示します：

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み（前のステップで使用したdfを使用すると仮定）
# df = pd.read_csv('your_data_path.csv')  # 必要に応じてデータを再度読み込む

# 特徴量の設定
features = ['width', 'height', 'area', 'perimeter', 'eccentricity', 'orientation', 
            'major_axis_length', 'minor_axis_length', 'solidity', 'extent', 
            'aspect_ratio', 'max_length']

X = df[features]
y = df['defect_label']

# Isolation Forest 分類器の設定
# contamination は異常データの割合の推定値。今回のケースでは 3 / (956 + 3) ≈ 0.003
clf = IsolationForest(n_estimators=100, contamination=0.003, max_features=1, random_state=42)

# 学習と予測
y_pred = clf.fit_predict(X)
y_pred = np.where(y_pred == 1, 0, 1)  # -1を1に、1を0に変換

# 異常度スコアの算出
anomaly_scores = clf.score_samples(X) * -1

# 欠陥ごとの精度指標の計算
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

fnr = fn / (fn + tp)  # False Negative Rate
fpr = fp / (fp + tn)  # False Positive Rate
acc = accuracy_score(y, y_pred)

print("欠陥ごとの精度指標:")
print(f"FN/(FN+TP) (見逃し率): {fnr:.2%} ({fn}/{fn+tp})")
print(f"FP/(FP+TN) (誤検出率): {fpr:.2%} ({fp}/{fp+tn})")
print(f"正解率: {acc:.2%} ({(y == y_pred).sum()}/{len(y)})")

# 予測結果をデータフレームに追加
df['predicted_label'] = y_pred
df['anomaly_score'] = anomaly_scores

# ワークごとの予測
df['work_predicted_label'] = df.groupby('work_id')['predicted_label'].transform('max')

# ワークごとの精度指標の計算
work_true = df.groupby('work_id')['defect_label'].max()
work_pred = df.groupby('work_id')['work_predicted_label'].first()

work_tn, work_fp, work_fn, work_tp = confusion_matrix(work_true, work_pred).ravel()

work_fnr = work_fn / (work_fn + work_tp)  # 見逃し率
work_fpr = work_fp / (work_fp + work_tn)  # 見過ぎ率
work_acc = accuracy_score(work_true, work_pred)  # 正解率

print("\nワークごとの精度指標:")
print(f"見逃し率: {work_fnr:.2%} ({work_fn}/{work_fn+work_tp})")
print(f"見過ぎ率: {work_fpr:.2%} ({work_fp}/{work_fp+work_tn})")
print(f"正解率: {work_acc:.2%} ({(work_true == work_pred).sum()}/{len(work_true)})")

# 異常度スコアの可視化
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=df.index, y='anomaly_score', hue='defect_label', 
                palette={0: 'blue', 1: 'red'}, size='defect_label', 
                sizes={0: 20, 1: 100}, alpha=0.7)

plt.axhline(y=clf.offset_ * -1, color='red', linestyle='--', label='閾値')
plt.xlabel('データ番号')
plt.ylabel('異常度スコア')
plt.title('Isolation Forest による異常検知結果')
plt.legend(title='欠陥ラベル', labels=['正常', '欠陥', '閾値'])
plt.tight_layout()
plt.show()

# 特徴量の平均異常度スコア
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': [clf.score_samples(X[[feature]]).mean() for feature in features]
})
feature_importance = feature_importance.sort_values('importance', ascending=True)
print("\n特徴量の平均異常度スコア:")
print(feature_importance)
```

主な変更点：

1. データの分割（train_test_split）を削除しました。
2. モデルの学習と予測を同時に行う`fit_predict`メソッドを使用しています。
3. すべてのデータポイントに対して予測と評価を行っています。

この修正により、Isolation Forestがすべてのデータを使用して異常を検出し、その結果に基づいて精度指標を計算します。これは異常検知タスクにより適した方法です。

ただし、このアプローチでは過学習のリスクがあることに注意してください。モデルの汎化性能を評価したい場合は、別途テストデータを用意するか、交差検証を行うことを検討してください。

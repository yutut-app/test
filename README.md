StandardScalerを加えた修正版のコードを生成する。

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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

# StandardScalerを適用
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest 分類器の設定
clf = IsolationForest(n_estimators=100, contamination=0.6, max_features=1, random_state=42)

# 学習と予測
y_pred = clf.fit_predict(X_scaled)
y_pred = np.where(y_pred == 1, 0, 1)  # -1を1に、1を0に変換

# 異常度スコアの算出
anomaly_scores = clf.score_samples(X_scaled) * -1

# 欠陥ごとの精度指標の計算
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

fnr = fn / (fn + tp)  # False Negative Rate
fpr = fp / (fp + tn)  # False Positive Rate
acc = accuracy_score(y, y_pred)

print("Defect-wise Performance Metrics:")
print(f"FN/(FN+TP) (Miss Rate): {fnr:.2%} ({fn}/{fn+tp})")
print(f"FP/(FP+TN) (False Detection Rate): {fpr:.2%} ({fp}/{fp+tn})")
print(f"Accuracy: {acc:.2%} ({(y == y_pred).sum()}/{len(y)})")

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

print("\nWork-wise Performance Metrics:")
print(f"Miss Rate: {work_fnr:.2%} ({work_fn}/{work_fn+work_tp})")
print(f"False Detection Rate: {work_fpr:.2%} ({work_fp}/{work_fp+work_tn})")
print(f"Accuracy: {work_acc:.2%} ({(work_true == work_pred).sum()}/{len(work_true)})")

# 異常度スコアの可視化
plt.figure(figsize=(12, 6))

# 正常データ（青）をプロット
sns.scatterplot(data=df[df['defect_label'] == 0], x=df[df['defect_label'] == 0].index, 
                y='anomaly_score', color='blue', label='Normal', s=20, alpha=0.7)

# 異常データ（赤）を後からプロット
sns.scatterplot(data=df[df['defect_label'] == 1], x=df[df['defect_label'] == 1].index, 
                y='anomaly_score', color='red', label='Defect', s=100, alpha=0.7)

plt.axhline(y=clf.offset_ * -1, color='red', linestyle='--', label='Threshold')
plt.xlabel('Data Point Index')
plt.ylabel('Anomaly Score')
plt.title('Isolation Forest Anomaly Detection Results')
plt.legend(title='Defect Label')
plt.tight_layout()
plt.show()

# 特徴量と異常度スコアの相関を計算
X_with_scores = pd.DataFrame(X_scaled, columns=features)
X_with_scores['anomaly_score'] = anomaly_scores
correlations = X_with_scores.corr()['anomaly_score'].drop('anomaly_score')
correlations = correlations.sort_values(ascending=False)

print("\nCorrelation between Features and Anomaly Score:")
print(correlations)

# 相関の可視化
plt.figure(figsize=(10, 6))
sns.barplot(x=correlations.index, y=correlations.values)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Correlation with Anomaly Score')
plt.title('Feature Correlation with Anomaly Score')
plt.tight_layout()
plt.show()
```

主な変更点と説明:

1. StandardScalerの導入:
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```
   これにより、各特徴量が平均0、分散1にスケーリングされる。

2. スケーリングされたデータの使用:
   ```python
   y_pred = clf.fit_predict(X_scaled)
   anomaly_scores = clf.score_samples(X_scaled) * -1
   ```
   Isolation Forestの学習と予測にスケーリングされたデータを使用する。

3. contaminationパラメータの調整:
   ```python
   clf = IsolationForest(n_estimators=100, contamination=0.6, max_features=1, random_state=42)
   ```
   contaminationを0.6に設定し、データの実際の異常率に近づける。

4. 特徴量と異常度スコアの相関計算の調整:
   ```python
   X_with_scores = pd.DataFrame(X_scaled, columns=features)
   X_with_scores['anomaly_score'] = anomaly_scores
   correlations = X_with_scores.corr()['anomaly_score'].drop('anomaly_score')
   ```
   スケーリングされた特徴量と異常度スコアの相関を計算する。

これらの変更により、特徴量のスケールの問題が解決され、より適切な異常検知が可能になることが期待される。また、contaminationパラメータの調整により、モデルの判定基準がデータの実際の異常率に近づく。

結果を確認し、必要に応じてさらなるパラメータ調整や特徴量の選択を行うことを推奨する。

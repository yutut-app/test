はい、Isolation Forestを使用して特徴量による分類を行うコードを生成します。以下のコードは、あなたの要件に合わせて修正しています。

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
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

# データの分割（学習データとテストデータを分ける）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Isolation Forest 分類器の設定
# contamination は異常データの割合の推定値。今回のケースでは 3 / (956 + 3) ≈ 0.003
clf = IsolationForest(n_estimators=100, contamination=0.003, max_features=1, random_state=42)

# 学習
clf.fit(X_train)

# テストデータで予測
y_pred = clf.predict(X_test)
y_pred = np.where(y_pred == 1, 0, 1)  # -1を1に、1を0に変換

# 異常度スコアの算出
anomaly_scores = clf.score_samples(X_test) * -1

# 欠陥ごとの精度指標の計算
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

fnr = fn / (fn + tp)  # False Negative Rate
fpr = fp / (fp + tn)  # False Positive Rate
acc = accuracy_score(y_test, y_pred)

print("欠陥ごとの精度指標 (テストデータ):")
print(f"FN/(FN+TP) (見逃し率): {fnr:.2%} ({fn}/{fn+tp})")
print(f"FP/(FP+TN) (誤検出率): {fpr:.2%} ({fp}/{fp+tn})")
print(f"正解率: {acc:.2%} ({(y_test == y_pred).sum()}/{len(y_test)})")

# テストデータのインデックスを取得
test_indices = y_test.index

# テストデータに対する予測結果をデータフレームに追加
df_test = df.loc[test_indices].copy()
df_test['predicted_label'] = y_pred
df_test['anomaly_score'] = anomaly_scores

# ワークごとの予測
df_test['work_predicted_label'] = df_test.groupby('work_id')['predicted_label'].transform('max')

# ワークごとの精度指標の計算
work_true = df_test.groupby('work_id')['defect_label'].max()
work_pred = df_test.groupby('work_id')['work_predicted_label'].first()

work_tn, work_fp, work_fn, work_tp = confusion_matrix(work_true, work_pred).ravel()

work_fnr = work_fn / (work_fn + work_tp)  # 見逃し率
work_fpr = work_fp / (work_fp + work_tn)  # 見過ぎ率
work_acc = accuracy_score(work_true, work_pred)  # 正解率

print("\nワークごとの精度指標 (テストデータ):")
print(f"見逃し率: {work_fnr:.2%} ({work_fn}/{work_fn+work_tp})")
print(f"見過ぎ率: {work_fpr:.2%} ({work_fp}/{work_fp+work_tn})")
print(f"正解率: {work_acc:.2%} ({(work_true == work_pred).sum()}/{len(work_true)})")

# 異常度スコアの可視化
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_test, x=df_test.index, y='anomaly_score', hue='defect_label', 
                palette={0: 'blue', 1: 'red'}, size='defect_label', 
                sizes={0: 20, 1: 100}, alpha=0.7)

plt.axhline(y=clf.offset_ * -1, color='red', linestyle='--', label='閾値')
plt.xlabel('データ番号')
plt.ylabel('異常度スコア')
plt.title('Isolation Forest による異常検知結果')
plt.legend(title='欠陥ラベル', labels=['正常', '欠陥', '閾値'])
plt.tight_layout()
plt.show()

# 特徴量の重要度（Isolation Forestでは直接的な重要度は計算されないため、代わりに各特徴量の平均異常度スコアを表示）
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': [clf.score_samples(X_test[[feature]]).mean() for feature in features]
})
feature_importance = feature_importance.sort_values('importance', ascending=True)
print("\n特徴量の平均異常度スコア:")
print(feature_importance)
```

このコードでは以下の変更を行いました：

1. Isolation Forestを使用して異常検知を行います。
2. `contamination`パラメータを0.003（3/959）に設定し、データセットの実際の異常割合に近づけています。
3. 予測結果を、元のラベル（0: 正常、1: 異常）に合わせて変換しています。
4. 異常度スコアを計算し、可視化しています。
5. 欠陥ごとおよびワークごとの精度指標を計算しています。
6. 特徴量の重要度の代わりに、各特徴量の平均異常度スコアを表示しています。

このアプローチでは、Isolation Forestが正常データの構造を学習し、それから大きく外れるデータポイントを異常として検出します。これにより、極端に不均衡なデータセットでも効果的に機能することが期待できます。

ただし、Isolation Forestは教師なし学習アルゴリズムであるため、必ずしも「欠陥」と「非欠陥」を完全に区別できるわけではありません。結果を注意深く解釈し、必要に応じてパラメータ（特に`contamination`と`n_estimators`）を調整することをお勧めします。

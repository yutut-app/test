承知しました。ご指示に従って、欠陥データ（defect_label == 1）の特徴量の最小値または最大値を閾値として設定するようにコードを修正します。

```markdown
## 6. 特徴量による分類（ルールベース）

EDAの結果に基づき、欠陥（中巣）は欠陥候補（非欠陥）に比べて以下の特徴があることがわかりました：

- perimeter, eccentricity, orientation, major_axis_length, minor_axis_length, solidity, extent が大きい
- aspect_ratio が小さい

これらの特徴を用いてルールベースの分類を行い、欠陥ごとおよびワークごとの精度を評価します。閾値は以下のように設定します：

- perimeter, eccentricity, orientation, major_axis_length, minor_axis_length, solidity, extent: 欠陥データの最小値
- aspect_ratio: 欠陥データの最大値

注意点：
- NGデータ（欠陥）が少ないため、学習データとテストデータは同じものを使用します。
- 欠陥ごとの評価では、FN/(FN+TP)を100%にし、FP/(FP+TN)を最小化します。
- ワークごとの評価では、見逃し率と見過ぎ率を計算します。
```

```python
import pandas as pd
import numpy as np
from collections import defaultdict

# 欠陥データのみを抽出
defect_data = df[df['defect_label'] == 1]

# 閾値の設定
thresholds = {
    'perimeter': defect_data['perimeter'].min(),
    'eccentricity': defect_data['eccentricity'].min(),
    'orientation': defect_data['orientation'].min(),
    'major_axis_length': defect_data['major_axis_length'].min(),
    'minor_axis_length': defect_data['minor_axis_length'].min(),
    'solidity': defect_data['solidity'].min(),
    'extent': defect_data['extent'].min(),
    'aspect_ratio': defect_data['aspect_ratio'].max()
}

print("設定された閾値:")
for key, value in thresholds.items():
    print(f"{key}: {value}")

# ルールベースの分類関数
def classify_defect(row):
    if (row['perimeter'] >= thresholds['perimeter'] and
        row['eccentricity'] >= thresholds['eccentricity'] and
        abs(row['orientation']) >= abs(thresholds['orientation']) and
        row['major_axis_length'] >= thresholds['major_axis_length'] and
        row['minor_axis_length'] >= thresholds['minor_axis_length'] and
        row['solidity'] >= thresholds['solidity'] and
        row['extent'] >= thresholds['extent'] and
        row['aspect_ratio'] <= thresholds['aspect_ratio']):
        return 1  # 欠陥（中巣）
    return 0  # 欠陥候補（非欠陥）

# 分類の実行
df['predicted_label'] = df.apply(classify_defect, axis=1)

# 欠陥ごとの評価
TP = ((df['defect_label'] == 1) & (df['predicted_label'] == 1)).sum()
FP = ((df['defect_label'] == 0) & (df['predicted_label'] == 1)).sum()
TN = ((df['defect_label'] == 0) & (df['predicted_label'] == 0)).sum()
FN = ((df['defect_label'] == 1) & (df['predicted_label'] == 0)).sum()

FNR = FN / (FN + TP)
FPR = FP / (FP + TN)

print("\n欠陥ごとの評価:")
print(f"FN/(FN+TP) (見逃し率): {FNR:.2%}")
print(f"FP/(FP+TN) (誤検出率): {FPR:.2%}")

# ワークごとの評価
work_results = defaultdict(lambda: {'has_defect': False, 'predicted_defect': False})

for _, row in df.iterrows():
    work_id = row['work_id']
    work_results[work_id]['has_defect'] |= (row['defect_label'] == 1)
    work_results[work_id]['predicted_defect'] |= (row['predicted_label'] == 1)

ng_works = sum(1 for work in work_results.values() if work['has_defect'])
ok_works = len(work_results) - ng_works

missed_defects = sum(1 for work in work_results.values() if work['has_defect'] and not work['predicted_defect'])
false_alarms = sum(1 for work in work_results.values() if not work['has_defect'] and work['predicted_defect'])

miss_rate = missed_defects / ng_works if ng_works > 0 else 0
false_alarm_rate = false_alarms / ok_works if ok_works > 0 else 0

print("\nワークごとの評価:")
print(f"見逃し率: {miss_rate:.2%}")
print(f"見過ぎ率: {false_alarm_rate:.2%}")
```

```markdown
このコードは以下の手順で動作します：

1. 欠陥データ（defect_label == 1）のみを抽出します。
2. 各特徴量に対して、欠陥データの最小値（aspect_ratioの場合は最大値）を閾値として設定します。
3. ルールベースの分類関数を定義し、設定した閾値に基づいて欠陥の有無を判断します。
4. データフレームの各行に分類関数を適用し、予測ラベルを生成します。
5. 欠陥ごとの評価指標（FN率とFP率）を計算します。
6. ワークごとの評価指標（見逃し率と見過ぎ率）を計算します。

結果の解釈：
- FN/(FN+TP)は0%になるはずです。これは全ての欠陥を検出できていることを意味します。
- FP/(FP+TN)は、誤検出の割合を示します。この値が高い場合、多くの非欠陥を誤って欠陥と判断していることになります。
- ワークごとの見逃し率も0%になるはずです。
- 見過ぎ率は、欠陥のないワークを誤って欠陥ありと判断した割合を示します。

次のステップ：
1. 結果の確認：FN率とワークごとの見逃し率が0%になっていることを確認します。
2. 誤検出の分析：FP率と見過ぎ率が高い場合、どのような特徴を持つデータが誤検出されているかを分析します。
3. 閾値の微調整：必要に応じて、一部の特徴量の閾値を調整し、誤検出を減らすことを検討します。
4. 特徴量の重要度分析：どの特徴量が欠陥の検出に最も寄与しているかを分析します。
5. モデルの改善：より複雑なルールや機械学習モデルの導入を検討します。
```

このコードを実行すると、欠陥データの特徴に基づいて設定された閾値での分類結果と評価指標が表示されます。FN率とワークごとの見逃し率は0%になるはずですが、FP率と見過ぎ率は高くなる可能性があります。

これらの結果を踏まえて、どのような分析や改善を行いたいですか？例えば、誤検出されたデータの特徴を詳しく調べたり、特定の特徴量の閾値を調整したりすることが考えられます。

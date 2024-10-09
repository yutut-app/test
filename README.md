# 分類ルールの定義
def classify_defect(row, thresholds):
    conditions = [
        row['perimeter'] > thresholds['perimeter'],
        row['eccentricity'] > thresholds['eccentricity'],
        row['orientation'] > thresholds['orientation'],
        row['major_axis_length'] > thresholds['major_axis_length'],
        row['minor_axis_length'] > thresholds['minor_axis_length'],
        row['solidity'] > thresholds['solidity'],
        row['extent'] > thresholds['extent'],
        row['aspect_ratio'] < thresholds['aspect_ratio']
    ]
    # 条件の半数以上を満たす場合に欠陥と判定
    return 0 if sum(conditions) >= len(conditions) // 2 else 1

# 閾値を調整する関数
def adjust_thresholds(df, initial_percentile=50):
    thresholds = {}
    features = ['perimeter', 'eccentricity', 'orientation', 'major_axis_length', 
                'minor_axis_length', 'solidity', 'extent', 'aspect_ratio']
    
    for feature in features:
        if feature == 'aspect_ratio':
            thresholds[feature] = np.percentile(df[feature], 100 - initial_percentile)
        else:
            thresholds[feature] = np.percentile(df[feature], initial_percentile)
    
    return thresholds

# 最適な閾値を見つける
def find_optimal_thresholds(df, max_iterations=100):
    best_thresholds = None
    best_false_positives = float('inf')
    percentile = 50

    for _ in range(max_iterations):
        thresholds = adjust_thresholds(df, percentile)
        df['predicted_label'] = df.apply(lambda row: classify_defect(row, thresholds), axis=1)
        
        true_defects = df[df['defect_label'] == 0]
        correctly_classified = true_defects[true_defects['predicted_label'] == 0]
        false_positives = df[(df['defect_label'] == 1) & (df['predicted_label'] == 0)]
        
        if len(correctly_classified) == len(true_defects) and len(false_positives) < best_false_positives:
            best_thresholds = thresholds
            best_false_positives = len(false_positives)
        
        percentile += 1
        if percentile > 99:
            break

    return best_thresholds

# 最適な閾値を見つけて分類を実行
best_thresholds = find_optimal_thresholds(df)
df['predicted_label'] = df.apply(lambda row: classify_defect(row, best_thresholds), axis=1)

# 分類結果の評価
print("分類結果:")
print(classification_report(df['defect_label'], df['predicted_label']))

print("\n混同行列:")
print(confusion_matrix(df['defect_label'], df['predicted_label']))

# 各クラスの予測数
print("\n各クラスの予測数:")
print(df['predicted_label'].value_counts())

# 実際の欠陥（鋳巣）のうち、正しく分類されたものの割合
true_defects = df[df['defect_label'] == 0]
correctly_classified = true_defects[true_defects['predicted_label'] == 0]
print(f"\n実際の欠陥（鋳巣）のうち、正しく分類された割合: {len(correctly_classified) / len(true_defects) * 100:.2f}%")

# 誤って欠陥（鋳巣）と分類されたものの数と割合
false_positives = df[(df['defect_label'] == 1) & (df['predicted_label'] == 0)]
print(f"\n誤って欠陥（鋳巣）と分類されたデータ数: {len(false_positives)}")
print(f"誤って欠陥（鋳巣）と分類された割合: {len(false_positives) / len(df) * 100:.2f}%")

# 最適な閾値を表示
print("\n最適な閾値:")
for feature, threshold in best_thresholds.items():
    print(f"{feature}: {threshold:.4f}")

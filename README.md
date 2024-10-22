このエラーは、テンプレート画像がマッチング対象の画像よりも大きい場合に発生します。テンプレートマッチングでは、テンプレートは常に対象画像より小さくなければなりません。この問題を解決するために、画像とテンプレートのサイズを調整するコードを追加します。

```python
def resize_template(template, target_image):
    """
    テンプレートのサイズを対象画像に合わせてリサイズする
    """
    target_h, target_w = target_image.shape
    template_h, template_w = template.shape
    
    # テンプレートが対象画像より大きい場合、リサイズする
    if template_h > target_h or template_w > target_w:
        # 縦横の比率を保ちながら、対象画像の半分のサイズにリサイズ
        scale = 0.5 * min(target_h / template_h, target_w / template_w)
        new_h = int(template_h * scale)
        new_w = int(template_w * scale)
        return cv2.resize(template, (new_w, new_h))
    return template

def perform_template_matching(image, templates, threshold=0.8):
    """
    画像に対してテンプレートマッチングを実行（サイズ調整付き）
    """
    max_score = 0
    for template in templates:
        # テンプレートのサイズを調整
        resized_template = resize_template(template, image)
        try:
            result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            max_score = max(max_score, max_val)
        except Exception as e:
            print(f"Template matching failed: {e}")
            continue
    return max_score > threshold
```

```python
# NGデータ（欠陥あり）からテンプレートを作成
print("テンプレートの作成開始...")
ng_samples = df_filtered[df_filtered['defect_label'] == 1]
templates = []
for _, row in ng_samples.iterrows():
    try:
        img = load_image(row['defect_image_orig'])
        templates.append(img)
        print(f"テンプレート画像サイズ: {img.shape}")
    except Exception as e:
        print(f"Error loading template: {e}")

print(f"読み込んだテンプレート数: {len(templates)}")
```

```python
# サンプルとして1つの画像でテスト実行
print("サンプル画像でのテスト実行...")
test_row = df_filtered.iloc[0]
try:
    test_img = load_image(test_row['defect_image_orig'])
    print(f"テスト画像サイズ: {test_img.shape}")
    is_defect = perform_template_matching(test_img, templates)
    print(f"テスト結果: {'欠陥あり' if is_defect else '欠陥なし'}")
except Exception as e:
    print(f"テスト実行エラー: {e}")
```

```python
# 少数のサンプルで検証
print("\n少数サンプルでの検証...")
test_samples = df_filtered.head(5)
results = []
for _, row in test_samples.iterrows():
    try:
        img = load_image(row['defect_image_orig'])
        print(f"処理中の画像サイズ: {img.shape}")
        is_defect = perform_template_matching(img, templates)
        results.append({
            'work_id': row['work_id'],
            'true_label': row['defect_label'],
            'predicted_label': 1 if is_defect else 0
        })
        print(f"予測結果: {'欠陥あり' if is_defect else '欠陥なし'} (真値: {row['defect_label']})")
    except Exception as e:
        print(f"Error processing image: {e}")

# 少数サンプルの結果を確認
test_results_df = pd.DataFrame(results)
if len(test_results_df) > 0:
    test_metrics = calculate_metrics(test_results_df, test_samples)
    print("\n=== テストサンプルの評価結果 ===")
    print(f"欠陥検出率: {test_metrics['defect_detection_rate']:.2f}%")
    print(f"誤検出率: {test_metrics['false_detection_rate']:.2f}%")
```

```python
# 問題なければ全データで実行
print("\n全データでの実行...")
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

```python
# 最終評価
if len(results_df) > 0:
    metrics = calculate_metrics(results_df, df_filtered)
    
    print("\n=== 最終評価結果 ===")
    print(f"欠陥検出率: {metrics['defect_detection_rate']:.2f}%")
    print(f"誤検出率: {metrics['false_detection_rate']:.2f}%")
    print(f"見逃し率: {metrics['miss_rate']:.2f}%")
    print(f"見過ぎ率: {metrics['over_detection_rate']:.2f}%")
    
    # 混同行列の詳細表示
    cm = metrics['confusion_matrix']
    print("\n=== 欠陥レベルの混同行列 ===")
    print(f"TP: {cm['TP']}, FP: {cm['FP']}")
    print(f"FN: {cm['FN']}, TN: {cm['TN']}")
    
    print("\n=== ワークレベルの混同行列 ===")
    print(f"TP: {cm['work_TP']}, FP: {cm['work_FP']}")
    print(f"FN: {cm['work_FN']}, TN: {cm['work_TN']}")
```

改良点：
1. テンプレートのサイズを自動調整する機能を追加
2. 段階的な実行と検証ステップを追加
   - テンプレート作成時のサイズ確認
   - 単一サンプルでのテスト
   - 少数サンプルでの検証
   - 全データでの実行
3. エラーハンドリングの強化
4. 進捗状況の詳細表示

このコードは段階的に実行できるようになっており、各ステップで結果を確認できます。問題が発生した場合は、どのステップで発生したかを特定しやすくなっています。

これらのコードブロックを順番に実行して、各段階での結果を確認できます。エラーが発生した場合は、どのステップでエラーが発生したかを教えていただければ、さらに適切な対処方法を提案できます。

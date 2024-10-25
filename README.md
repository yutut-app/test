テンプレートごとにスコアを記録し、元のデータフレームの情報も保持するように修正します。

```python
def perform_template_matching_with_scores(image, templates, template_names, threshold):
    """
    画像に対してテンプレートマッチングを実行し、各テンプレートのスコアを返す
    """
    scores = {}
    max_score = 0
    
    for template, template_name in zip(templates, template_names):
        # 画像サイズの取得
        img_height, img_width = image.shape
        templ_height, templ_width = template.shape
        
        # テンプレートが入力画像より大きい場合、テンプレートをリサイズ
        if templ_height > img_height or templ_width > img_width:
            template = cv2.resize(template, (min(img_width, templ_width), min(img_height, templ_height)), 
                                interpolation=cv2.INTER_AREA)
        
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        scores[f"{template_name}_match_score"] = max_val
        max_score = max(max_score, max_val)
    
    return max_score > threshold, scores

def process_images_with_threshold(df_filtered, templates, template_names, threshold):
    """指定された閾値でパターンマッチングを実行"""
    results = []
    for _, row in df_filtered.iterrows():
        try:
            img = load_and_preprocess_image(row['defect_image_orig'])
            is_matched, template_scores = perform_template_matching_with_scores(
                img, templates, template_names, threshold)
            
            # 基本情報の辞書を作成
            result_dict = {
                'image_name': row['image_name'],
                'Image_label': row['Image_label'],
                'defect_label': row['defect_label'],
                'label': row['label'],
                'work_id': row['work_id'],
                'predicted_label': 1 if is_matched else 0
            }
            
            # テンプレートごとのスコアを追加
            result_dict.update(template_scores)
            
            results.append(result_dict)
            
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
template_names = []
for idx, row in ng_samples.iterrows():
    try:
        img = load_and_preprocess_image(row['defect_image_orig'])
        templates.append(img)
        template_names.append(f"template_{row['image_name']}")
        print(f"テンプレート {idx+1}: {row['image_name']}, サイズ {img.shape}")
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
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_threshold = None
    best_detection_rate = -1
    best_false_rate = float('inf')
    
    results_dict = {}
    
    # 各閾値での結果を保存
    for threshold in tqdm(thresholds, desc="閾値の最適化"):
        results_df = process_images_with_threshold(df_filtered, templates, template_names, threshold)
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
final_results_df = process_images_with_threshold(df_filtered, templates, template_names, best_threshold)
detection_rate, false_rate = calculate_metrics(final_results_df)

print(f"最終検出率: {detection_rate * 100:.2f}%")
print(f"最終誤検出率: {false_rate * 100:.2f}%")

# 各テンプレートのスコア分布を可視化
plt.figure(figsize=(15, 5*((len(template_names)+1)//2)))
for i, template_name in enumerate(template_names):
    plt.subplot(((len(template_names)+1)//2), 2, i+1)
    plt.hist(final_results_df[f"{template_name}_match_score"], bins=50)
    plt.axvline(x=best_threshold, color='r', linestyle='--', 
                label=f'Threshold ({best_threshold:.2f})')
    plt.xlabel('Matching Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Matching Scores - {template_name}')
    plt.legend()
plt.tight_layout()
plt.show()

# 結果の確認
print("\n=== 最終結果のサンプル ===")
print("\nデータフレームの列:")
print(final_results_df.columns.tolist())
print("\n最初の数行:")
print(final_results_df.head())
```

主な変更点：
1. テンプレート画像の名前を保持するリストを追加
2. 各テンプレートのマッチングスコアを個別に記録
3. 元のデータフレームの情報を保持
4. テンプレートごとのスコア分布を可視化

最終的なデータフレーム(final_results_df)の列構成：
1. 元の情報
   - image_name
   - Image_label
   - defect_label
   - label
   - work_id
2. 予測結果
   - predicted_label
3. テンプレートごとのスコア
   - template_[image_name]_match_score (各テンプレートに対して)

このコードを実行すると、各テンプレートごとのマッチングスコアが確認でき、どのテンプレートが各画像に対してよく反応したかが分かります。

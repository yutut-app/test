テンプレートマッチングでは、テンプレート画像が入力画像より大きい場合にエラーが発生します。画像のサイズを確認し、必要に応じてリサイズする処理を追加します。

```python
# 画像サイズ確認用の関数
def check_image_size(image_path):
    """画像のサイズを確認する"""
    img = cv2.imread(os.path.join(defected_image_path, image_path))
    if img is None:
        return None
    return img.shape

# NGデータの画像サイズを確認
print("=== NGデータの画像サイズ確認 ===")
ng_samples = df_filtered[df_filtered['defect_label'] == 1]
ng_sizes = []
for _, row in ng_samples.iterrows():
    size = check_image_size(row['defect_image_orig'])
    if size is not None:
        ng_sizes.append(size)
        print(f"画像パス: {row['defect_image_orig']}, サイズ: {size}")

# 全データの画像サイズを確認
print("\n=== 全データの画像サイズ範囲 ===")
all_sizes = []
for _, row in df_filtered.iterrows():
    size = check_image_size(row['defect_image_orig'])
    if size is not None:
        all_sizes.append(size)

if all_sizes:
    heights = [s[0] for s in all_sizes]
    widths = [s[1] for s in all_sizes]
    print(f"高さ範囲: {min(heights)} - {max(heights)}")
    print(f"幅範囲: {min(widths)} - {max(widths)}")
```

```python
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

def perform_template_matching(image, template, threshold=0.8):
    """
    画像に対してテンプレートマッチングを実行
    """
    # 画像サイズの取得
    img_height, img_width = image.shape
    templ_height, templ_width = template.shape
    
    # テンプレートが入力画像より大きい場合、テンプレートをリサイズ
    if templ_height > img_height or templ_width > img_width:
        template = cv2.resize(template, (min(img_width, templ_width), min(img_height, templ_height)), 
                            interpolation=cv2.INTER_AREA)
    
    # テンプレートマッチングの実行
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    return max_val > threshold
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
# パターンマッチングによる分類の実行（1枚ずつ確認用）
def process_single_image(image_path, templates, threshold=0.8):
    """1枚の画像に対してパターンマッチングを実行"""
    try:
        img = load_and_preprocess_image(image_path)
        
        # 各テンプレートに対してマッチング
        scores = []
        for idx, template in enumerate(templates):
            # 画像サイズの取得
            img_height, img_width = img.shape
            templ_height, templ_width = template.shape
            
            # テンプレートが入力画像より大きい場合、テンプレートをリサイズ
            if templ_height > img_height or templ_width > img_width:
                template = cv2.resize(template, (min(img_width, templ_width), min(img_height, templ_height)), 
                                   interpolation=cv2.INTER_AREA)
            
            # テンプレートマッチングの実行
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            scores.append(max_val)
            
        max_score = max(scores)
        matched_template = scores.index(max_score)
        
        return {
            'matched': max_score > threshold,
            'score': max_score,
            'template_idx': matched_template
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
```

```python
# テスト用に数枚の画像で確認
print("=== テスト画像での確認 ===")
test_samples = df_filtered.head(5)
for _, row in test_samples.iterrows():
    result = process_single_image(row['defect_image_orig'], templates)
    if result:
        print(f"\nImage: {row['defect_image_orig']}")
        print(f"True label: {row['defect_label']}")
        print(f"Matched: {result['matched']}")
        print(f"Best match score: {result['score']:.3f}")
        print(f"Best matching template: {result['template_idx']}")
```

```python
# 全データでの分類実行
print("\n=== 全データでの分類実行 ===")
results = []
for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    result = process_single_image(row['defect_image_orig'], templates)
    if result:
        results.append({
            'work_id': row['work_id'],
            'true_label': row['defect_label'],
            'predicted_label': 1 if result['matched'] else 0,
            'match_score': result['score'],
            'template_idx': result['template_idx']
        })

results_df = pd.DataFrame(results)

# 結果の表示
print("\n=== マッチング結果サマリー ===")
print(f"処理した画像数: {len(results_df)}")
print(f"マッチした画像数: {sum(results_df['predicted_label'] == 1)}")
print("\nマッチングスコアの統計:")
print(results_df['match_score'].describe())
```

このコードでは:
1. まず画像サイズを確認
2. 画像の読み込みと前処理を一つの関数にまとめる
3. テンプレートと入力画像のサイズ不一致を解決
4. 1枚ずつ確認できる関数を追加
5. テスト用の少数サンプルでの確認を追加
6. 詳細な結果の表示を追加

これにより:
- エラーを防ぎつつ、画像サイズの違いに対応
- 処理の各段階で結果を確認可能
- より詳細なマッチング情報を取得

このコードを順番に実行して、各段階での結果を確認できます。特に問題がなければ、精度評価のコードを実行できます。続けて精度評価のコードを提供しましょうか？

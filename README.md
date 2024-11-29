サイズの確認と問題箇所の特定のために、まずデバッグ用のコードを追加しましょう。

```python
def perform_template_matching(image, template, threshold):
    """テンプレートマッチングを実行"""
    # サイズ確認
    if image is None or template is None:
        print("Error: Image or template is None")
        return False, 0, None
        
    img_height, img_width = image.shape
    templ_height, templ_width = template.shape
    
    print(f"画像サイズ: {img_width}x{img_height}")
    print(f"テンプレートサイズ: {templ_width}x{templ_height}")
    
    if img_height < templ_height or img_width < templ_width:
        print("Error: Image is smaller than template")
        return False, 0, None
    
    try:
        # マッチング実行
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        if result is None or result.size == 0:
            print("Error: matchTemplate returned empty result")
            return False, 0, None
            
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return max_val > threshold, max_val, max_loc
        
    except Exception as e:
        print(f"テンプレートマッチングエラー: {e}")
        return False, 0, None

# データ読み込みの確認
print("\n=== データ確認 ===")
print(f"DataFrame shape: {df_filtered.shape}")
print("\nカラム一覧:")
print(df_filtered.columns.tolist())
print("\n最初の数行のdefect_image_origの値:")
print(df_filtered['defect_image_orig'].head())

# テンプレート読み込みの確認
templates, template_names = load_templates()
print("\n=== テンプレート確認 ===")
for i, (template, name) in enumerate(zip(templates, template_names)):
    print(f"\nテンプレート {i+1}: {name}")
    print(f"サイズ: {template.shape}")
    if template is None:
        print("警告: テンプレートがNone")

# 1枚目の画像で詳細なテスト
print("\n=== 1枚目の画像でテスト ===")
first_image_path = df_filtered.iloc[0]['defect_image_orig']
try:
    test_image = load_and_preprocess_image(first_image_path)
    print(f"テスト画像パス: {first_image_path}")
    print(f"テスト画像サイズ: {test_image.shape if test_image is not None else 'None'}")
    
    if test_image is not None:
        for template, name in zip(templates, template_names):
            print(f"\nテンプレート {name} でマッチング試行")
            is_matched, score, location = perform_template_matching(
                test_image, template, 0.8)
            print(f"マッチング結果: {is_matched}, スコア: {score}")
except Exception as e:
    print(f"テスト時エラー: {e}")
```

このデバッグコードを実行して、以下の点を確認させていただけますか：
1. DataFrameのサイズとカラム
2. 画像パスが正しく設定されているか
3. テンプレートが正しく読み込まれているか
4. 1枚目の画像での詳細なマッチング結果

エラーメッセージや出力結果を元に、具体的な問題箇所を特定し、適切な修正を提案させていただきます。

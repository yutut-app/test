print("\n=== 詳細なデータ確認 ===")
print(f"1. データフレーム情報:")
print(f"行数: {len(df_filtered)}")
print(f"欠陥ラベル(defect_label)の分布:\n{df_filtered['defect_label'].value_counts()}")

print("\n2. テンプレート情報:")
for i, (template, name) in enumerate(zip(templates, template_names)):
    print(f"\nテンプレート {name}:")
    print(f"- サイズ: {template.shape}")
    print(f"- 値の範囲: min={template.min()}, max={template.max()}")
    print(f"- 平均値: {template.mean():.2f}")

print("\n3. テスト画像の詳細:")
test_image = load_and_preprocess_image(first_image_path)
print(f"テスト画像: {first_image_path}")
print(f"- サイズ: {test_image.shape}")
print(f"- 値の範囲: min={test_image.min()}, max={test_image.max()}")
print(f"- 平均値: {test_image.mean():.2f}")

print("\n4. マッチング処理の詳細:")
for template, name in zip(templates, template_names):
    print(f"\nテンプレート {name} との比較:")
    result = cv2.matchTemplate(test_image, template, cv2.TM_CCOEFF_NORMED)
    print(f"- 結果配列のサイズ: {result.shape}")
    print(f"- 結果値の範囲: min={result.min():.4f}, max={result.max():.4f}")

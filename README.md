# 最適な閾値での最終評価
print("\n=== 最適な閾値での最終評価 ===")
final_results_df = process_images_with_threshold(df_filtered, templates, template_names, best_threshold)
detection_rate, false_rate = calculate_metrics(final_results_df)

print(f"最終検出率: {detection_rate * 100:.2f}%")
print(f"最終誤検出率: {false_rate * 100:.2f}%")

# 各テンプレートのスコア分布を可視化
plt.figure(figsize=(15, 5 * ((len(template_names) + 1) // 2)))
for i, template_name in enumerate(template_names, 1):
    score_column = f"{template_name}_match_score"
    plt.subplot((len(template_names) + 1) // 2, 2, i)
    plt.hist(final_results_df[score_column], bins=50)
    plt.axvline(x=best_threshold, color='r', linestyle='--', 
                label=f'Threshold ({best_threshold:.2f})')
    plt.xlabel('Matching Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {template_name} Matching Scores')
    plt.legend()
plt.tight_layout()
plt.show()

# 結果の保存（オプション）
print("\n=== 最終結果の確認 ===")
print("\n元の列名:")
print(list(df_filtered.columns))
print("\n新しい列名:")
print(list(final_results_df.columns))

# テンプレートごとの最高スコアの統計
print("\n=== テンプレートごとの最高スコア統計 ===")
for template_name in template_names:
    score_column = f"{template_name}_match_score"
    print(f"\n{template_name}:")
    print(final_results_df[score_column].describe())

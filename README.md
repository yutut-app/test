はい、テンプレートフォルダから直接テンプレートを読み込むように修正します。

```python
# 必要なライブラリのインポート
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# テンプレート画像のパスを設定
template_dir = r"../data/input/template_matching_template"

def load_and_preprocess_image(image_path, is_template=False):
    """画像を読み込み、前処理を行う"""
    # テンプレート画像とそれ以外で読み込みパスを変える
    if is_template:
        full_path = image_path
    else:
        full_path = os.path.join(defected_image_path, image_path)
    
    img = cv2.imread(full_path)
    if img is None:
        raise ValueError(f"Failed to load image: {full_path}")
    
    # グレースケール化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def load_templates(template_dir):
    """テンプレートフォルダから全てのテンプレート画像を読み込む"""
    templates = []
    template_names = []
    
    # テンプレートディレクトリ内の全ての画像ファイルを取得
    template_files = [f for f in os.listdir(template_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print("=== テンプレート画像の読み込み ===")
    for template_file in template_files:
        try:
            template_path = os.path.join(template_dir, template_file)
            template = load_and_preprocess_image(template_path, is_template=True)
            templates.append(template)
            
            # ファイル名から拡張子を除いてテンプレート名とする
            template_name = os.path.splitext(template_file)[0]
            template_names.append(template_name)
            
            print(f"テンプレート '{template_name}': サイズ {template.shape}")
        except Exception as e:
            print(f"Error loading template {template_file}: {e}")
    
    print(f"\n読み込んだテンプレート数: {len(templates)}")
    return templates, template_names

def perform_template_matching(image, templates, template_names, threshold):
    """
    画像に対してテンプレートマッチングを実行
    """
    template_scores = {}
    is_matched = False
    
    for template, template_name in zip(templates, template_names):
        # 画像サイズの取得
        img_height, img_width = image.shape
        templ_height, templ_width = template.shape
        
        # テンプレートが入力画像より大きい場合、テンプレートをリサイズ
        if templ_height > img_height or templ_width > img_width:
            template = cv2.resize(template, 
                               (min(img_width, templ_width), min(img_height, templ_height)), 
                               interpolation=cv2.INTER_AREA)
        
        # テンプレートマッチングの実行
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        # スコアを保存
        template_scores[f"{template_name}_match_score"] = max_val
        
        # 閾値を超えるスコアがあれば、マッチしたとみなす
        if max_val > threshold:
            is_matched = True
    
    return is_matched, template_scores

def process_images_with_threshold(df_filtered, templates, template_names, threshold):
    """指定された閾値でパターンマッチングを実行"""
    results = []
    for _, row in tqdm(df_filtered.iterrows(), desc="Processing images"):
        try:
            img = load_and_preprocess_image(row['defect_image_orig'])
            is_matched, template_scores = perform_template_matching(img, templates, template_names, threshold)
            
            # 元のデータフレームの行をコピー
            result_row = row.copy()
            # 予測結果とスコアを追加
            result_row['predicted_label'] = 1 if is_matched else 0
            for score_name, score in template_scores.items():
                result_row[score_name] = score
            
            results.append(result_row)
            
        except Exception as e:
            print(f"Error processing image {row['defect_image_orig']}: {e}")
            continue
    
    return pd.DataFrame(results)

def calculate_metrics(results_df):
    """精度指標を計算する"""
    # 欠陥ごとの評価
    TP = sum((results_df['true_label'] == 1) & (results_df['predicted_label'] == 1))
    FP = sum((results_df['true_label'] == 0) & (results_df['predicted_label'] == 1))
    FN = sum((results_df['true_label'] == 1) & (results_df['predicted_label'] == 0))
    TN = sum((results_df['true_label'] == 0) & (results_df['predicted_label'] == 0))
    
    defect_detection_rate = TP / (FN + TP) if (FN + TP) > 0 else 0
    false_detection_rate = FP / (TN + FP) if (TN + FP) > 0 else 0
    
    return defect_detection_rate, false_detection_rate
```

```python
# テンプレートの読み込み
templates, template_names = load_templates(template_dir)

# テスト用に少数のサンプルで実行
print("\n=== テストサンプルでの実行 ===")
test_df = df_filtered.head(3)
test_results = process_images_with_threshold(test_df, templates, template_names, threshold=0.8)
print("\nテスト結果の最初の行:")
print(test_results.iloc[0][['predicted_label'] + [f"{name}_match_score" for name in template_names]])
```

```python
def optimize_threshold():
    """
    defect_detection_rateを最大化し、
    同じdefect_detection_rateの場合はfalse_detection_rateを最小化する閾値を探索
    """
    thresholds = np.arange(0.1, 1.0, 0.05)  # 探索する閾値の範囲
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
        
        print(f"\n閾値 {threshold:.2f}:")
        print(f"検出率: {detection_rate * 100:.2f}%")
        print(f"誤検出率: {false_rate * 100:.2f}%")
    
    return best_threshold, results_dict

# 最適化の実行
print("\n=== 閾値の最適化を開始 ===")
best_threshold, all_results = optimize_threshold()
```

```python
# 最適な閾値での最終評価
print("\n=== 最適な閾値での最終評価 ===")
final_results_df = process_images_with_threshold(df_filtered, templates, template_names, best_threshold)
detection_rate, false_rate = calculate_metrics(final_results_df)

print(f"最適な閾値: {best_threshold:.2f}")
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

# 詳細な結果の表示
print("\n=== テンプレートごとのスコア統計 ===")
for template_name in template_names:
    score_column = f"{template_name}_match_score"
    print(f"\n{template_name}:")
    print(final_results_df[score_column].describe())

# 結果の保存（オプション）
print("\n=== 予測結果の内訳 ===")
print("\n予測結果とラベルのクロス集計:")
print(pd.crosstab(final_results_df['true_label'], final_results_df['predicted_label'],
                  margins=True))
```

主な変更点：
1. テンプレートフォルダから直接テンプレートを読み込むように変更
2. テンプレート名をファイル名から自動的に設定
3. 画像読み込み時のパス処理を改善
4. より詳細な進捗と結果の表示を追加

使用方法：
1. テンプレートの読み込みコードを実行
2. テストサンプルで動作確認
3. 閾値の最適化を実行
4. 最終評価と結果の可視化を実行

注意点：
- テンプレートフォルダ内の画像ファイルは.png、.jpg、または.jpegである必要があります
- ファイル名がテンプレート名として使用されます
- エラーが発生した場合は詳細なエラーメッセージが表示されます

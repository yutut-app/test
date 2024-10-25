コードが長くなるため、複数のメッセージに分けて送信します。まず前半部分です。

```python
# 必要なライブラリのインポート
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# パスの設定
defected_data_path = r"../data/output/defect_data"
defected_image_path = r"../data/output/defect_data/defect_images"
template_path = r"../data/input/template_matching_template"
defected_csv = "defects_data.csv"
defected_csv_path = os.path.join(defected_data_path, defected_csv)

# 画像サイズ確認用の関数
def check_image_size(image_path):
    """画像のサイズを確認する"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return img.shape

def load_and_preprocess_image(image_path, is_template=False):
    """
    画像を読み込み、前処理を行う
    
    Parameters:
    image_path (str): 画像のパス
    is_template (bool): テンプレート画像かどうか
    
    Returns:
    numpy.ndarray: 前処理済みのグレースケール画像
    """
    # テンプレート画像と入力画像でパスの扱いを分ける
    if is_template:
        img = cv2.imread(image_path)
    else:
        img = cv2.imread(os.path.join(defected_image_path, image_path))
        
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # グレースケール化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def load_templates():
    """
    テンプレートフォルダから全てのテンプレート画像を読み込む
    
    Returns:
    tuple: (templates, template_names)
    templates: 前処理済みテンプレート画像のリスト
    template_names: テンプレート名のリスト
    """
    templates = []
    template_names = []
    
    # テンプレートフォルダ内の全ての画像ファイルを取得
    template_files = [f for f in os.listdir(template_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print("=== テンプレート画像の読み込み ===")
    for template_file in template_files:
        try:
            template_full_path = os.path.join(template_path, template_file)
            img = load_and_preprocess_image(template_full_path, is_template=True)
            templates.append(img)
            
            # ファイル名から拡張子を除いてテンプレート名とする
            template_name = os.path.splitext(template_file)[0]
            template_names.append(template_name)
            
            print(f"テンプレート {template_name}: サイズ {img.shape}")
        except Exception as e:
            print(f"Error loading template {template_file}: {e}")
            continue
    
    print(f"\n読み込んだテンプレート数: {len(templates)}")
    return templates, template_names

def perform_template_matching(image, template, threshold):
    """
    1つのテンプレートに対してテンプレートマッチングを実行
    
    Parameters:
    image (numpy.ndarray): 入力画像
    template (numpy.ndarray): テンプレート画像
    threshold (float): マッチング閾値
    
    Returns:
    tuple: (is_matched, score)
    is_matched: 閾値を超えるマッチングが見つかったかどうか
    score: 最大マッチングスコア
    """
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
    
    return max_val > threshold, max_val

def process_single_image(image_path, templates, template_names, threshold):
    """
    1枚の画像に対して全てのテンプレートでマッチングを実行
    
    Parameters:
    image_path (str): 画像のパス
    templates (list): テンプレート画像のリスト
    template_names (list): テンプレート名のリスト
    threshold (float): マッチング閾値
    
    Returns:
    dict: マッチング結果と各テンプレートのスコア
    """
    try:
        img = load_and_preprocess_image(image_path)
        template_scores = {}
        is_matched = False
        
        # 各テンプレートに対してマッチングを実行
        for template, template_name in zip(templates, template_names):
            matched, score = perform_template_matching(img, template, threshold)
            template_scores[f"{template_name}_match_score"] = score
            if matched:
                is_matched = True
        
        return {
            'is_matched': is_matched,
            'template_scores': template_scores
        }
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
```

パターンマッチングの実装の続きです。

```python
def process_images_with_threshold(df_filtered, templates, template_names, threshold):
    """
    指定された閾値で全画像に対してパターンマッチングを実行
    
    Parameters:
    df_filtered (pandas.DataFrame): 処理対象のデータフレーム
    templates (list): テンプレート画像のリスト
    template_names (list): テンプレート名のリスト
    threshold (float): マッチング閾値
    
    Returns:
    pandas.DataFrame: 元のデータフレームにマッチング結果を追加したもの
    """
    results = []
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="画像処理中"):
        try:
            # 元のデータフレームの行をコピー
            result_row = row.copy()
            
            # 画像処理の実行
            matching_result = process_single_image(
                row['defect_image_orig'], 
                templates, 
                template_names, 
                threshold
            )
            
            if matching_result is not None:
                # 予測結果の追加
                result_row['predicted_label'] = 1 if matching_result['is_matched'] else 0
                
                # 各テンプレートのスコアを追加
                for score_name, score in matching_result['template_scores'].items():
                    result_row[score_name] = score
                
                results.append(result_row)
            
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    return pd.DataFrame(results)

def calculate_metrics(results_df):
    """
    精度指標を計算する
    
    Parameters:
    results_df (pandas.DataFrame): 予測結果を含むデータフレーム
    
    Returns:
    tuple: (defect_detection_rate, false_detection_rate)
    """
    # 欠陥ごとの評価
    TP = sum((results_df['defect_label'] == 1) & (results_df['predicted_label'] == 1))
    FP = sum((results_df['defect_label'] == 0) & (results_df['predicted_label'] == 1))
    FN = sum((results_df['defect_label'] == 1) & (results_df['predicted_label'] == 0))
    TN = sum((results_df['defect_label'] == 0) & (results_df['predicted_label'] == 0))
    
    defect_detection_rate = TP / (FN + TP) if (FN + TP) > 0 else 0
    false_detection_rate = FP / (TN + FP) if (TN + FP) > 0 else 0
    
    return defect_detection_rate, false_detection_rate

def optimize_threshold():
    """
    defect_detection_rateを最大化し、
    同じdefect_detection_rateの場合はfalse_detection_rateを最小化する閾値を探索
    
    Returns:
    tuple: (best_threshold, results_dict)
    """
    thresholds = np.arange(0.1, 1.0, 0.05)  # 探索する閾値の範囲
    best_threshold = None
    best_detection_rate = -1
    best_false_rate = float('inf')
    
    results_dict = {}
    
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

# メイン処理の実行
print("=== データの読み込み ===")
df = pd.read_csv(defected_csv_path)
df_filtered = df[df['work_predict_label'] == 1].copy()
print(f"処理対象データ数: {len(df_filtered)}")

# テンプレートの読み込み
templates, template_names = load_templates()

# 閾値の最適化
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
# 最適な閾値での最終評価と結果の可視化
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
    
    # 欠陥/非欠陥でスコアを分けてプロット
    defect_scores = final_results_df[final_results_df['defect_label'] == 1][score_column]
    non_defect_scores = final_results_df[final_results_df['defect_label'] == 0][score_column]
    
    plt.hist(defect_scores, bins=20, alpha=0.5, label='Defect', color='red')
    plt.hist(non_defect_scores, bins=20, alpha=0.5, label='Non-defect', color='blue')
    
    plt.axvline(x=best_threshold, color='g', linestyle='--', 
                label=f'Threshold ({best_threshold:.2f})')
    plt.xlabel('Matching Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {template_name} Matching Scores')
    plt.legend()

plt.tight_layout()
plt.show()

# テンプレートごとの性能統計
print("\n=== テンプレートごとの統計 ===")
for template_name in template_names:
    score_column = f"{template_name}_match_score"
    print(f"\n{template_name}:")
    print("欠陥データのスコア:")
    print(final_results_df[final_results_df['defect_label'] == 1][score_column].describe())
    print("\n非欠陥データのスコア:")
    print(final_results_df[final_results_df['defect_label'] == 0][score_column].describe())

# 結果の保存（必要に応じて）
# final_results_df.to_csv('template_matching_results.csv', index=False)
```

この実装の主な特徴：
1. テンプレートフォルダから自動的に全テンプレートを読み込み
2. 各テンプレートに対する個別のスコアを保持
3. 欠陥検出率を最大化し、同率の場合は誤検出率を最小化する閾値の最適化
4. 詳細な結果の可視化と統計情報の提供
5. エラーハンドリングとログ出力の充実

使用方法：
1. テンプレート画像を`template_path`で指定したフォルダに配置
2. コードを順番に実行
3. 最適な閾値と各テンプレートの性能を確認
4. 必要に応じて閾値の探索範囲を調整

注意点：
- テンプレート画像は.png、.jpg、.jpeg形式に対応
- テンプレート名はファイル名（拡張子を除く）から自動生成
- メモリ使用量が大きくなる可能性があるため、必要に応じてバッチ処理を検討

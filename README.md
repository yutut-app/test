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

def load_and_preprocess_image(image_path, is_template=False):
    """
    画像を読み込み、前処理を行う
    
    Parameters:
    image_path (str): 画像のパス
    is_template (bool): テンプレート画像かどうか
    
    Returns:
    numpy.ndarray: 前処理済みのグレースケール画像
    """
    if is_template:
        img = cv2.imread(image_path)
    else:
        img = cv2.imread(os.path.join(defected_image_path, image_path))
        
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def load_templates():
    """
    テンプレートフォルダから全てのテンプレート画像を読み込む
    """
    templates = []
    template_names = []
    
    template_files = [f for f in os.listdir(template_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print("=== テンプレート画像の読み込み ===")
    for template_file in template_files:
        try:
            template_full_path = os.path.join(template_path, template_file)
            img = load_and_preprocess_image(template_full_path, is_template=True)
            templates.append(img)
            
            template_name = os.path.splitext(template_file)[0]
            template_names.append(template_name)
            
            print(f"テンプレート {template_name}: サイズ {img.shape}")
        except Exception as e:
            print(f"Error loading template {template_file}: {e}")
            continue
    
    print(f"\n読み込んだテンプレート数: {len(templates)}")
    return templates, template_names

def search_template_in_image(image, template, threshold):
    """
    画像内でテンプレートを探索する
    
    Parameters:
    image (numpy.ndarray): 探索対象の画像
    template (numpy.ndarray): テンプレート画像
    threshold (float): マッチング閾値
    
    Returns:
    tuple: (is_matched, max_score)
    """
    # 画像サイズの確認
    ih, iw = image.shape
    th, tw = template.shape
    
    # テンプレートが欠陥候補画像より大きい場合はスキップ
    if th > ih or tw > iw:
        return False, 0.0
    
    # テンプレートマッチング実行
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    # 閾値との比較
    is_matched = max_val > threshold
    
    return is_matched, max_val

def process_single_image(image_path, templates, template_names, threshold):
    """
    1枚の画像に対して全てのテンプレートでマッチングを実行
    """
    try:
        img = load_and_preprocess_image(image_path)
        template_scores = {}
        is_matched = False
        highest_score = 0
        
        for template, template_name in zip(templates, template_names):
            matched, score = search_template_in_image(img, template, threshold)
            template_scores[f"{template_name}_match_score"] = score
            
            if matched:
                is_matched = True
                highest_score = max(highest_score, score)
        
        return {
            'is_matched': is_matched,
            'template_scores': template_scores,
            'highest_score': highest_score
        }
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def process_images_with_threshold(df_filtered, templates, template_names, threshold):
    """
    指定された閾値で全画像に対してパターンマッチングを実行
    """
    results = []
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="画像処理中"):
        try:
            result_row = row.copy()
            
            matching_result = process_single_image(
                row['defect_image_orig'], 
                templates, 
                template_names, 
                threshold
            )
            
            if matching_result is not None:
                result_row['predicted_label'] = 1 if matching_result['is_matched'] else 0
                result_row['highest_match_score'] = matching_result['highest_score']
                
                for score_name, score in matching_result['template_scores'].items():
                    result_row[score_name] = score
                
                results.append(result_row)
            
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    return pd.DataFrame(results)

def calculate_metrics(results_df):
    """精度指標を計算する"""
    TP = sum((results_df['defect_label'] == 1) & (results_df['predicted_label'] == 1))
    FP = sum((results_df['defect_label'] == 0) & (results_df['predicted_label'] == 1))
    FN = sum((results_df['defect_label'] == 1) & (results_df['predicted_label'] == 0))
    TN = sum((results_df['defect_label'] == 0) & (results_df['predicted_label'] == 0))
    
    defect_detection_rate = TP / (FN + TP) if (FN + TP) > 0 else 0
    false_detection_rate = FP / (TN + FP) if (TN + FP) > 0 else 0
    
    return defect_detection_rate, false_detection_rate

def optimize_threshold():
    """閾値の最適化"""
    thresholds = np.arange(0.1, 1.0, 0.05)
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

# 最終評価
print("\n=== 最適な閾値での最終評価 ===")
final_results_df = process_images_with_threshold(df_filtered, templates, template_names, best_threshold)
detection_rate, false_rate = calculate_metrics(final_results_df)

print(f"最終検出率: {detection_rate * 100:.2f}%")
print(f"最終誤検出率: {false_rate * 100:.2f}%")

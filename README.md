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

続きは次のメッセージで送信します...

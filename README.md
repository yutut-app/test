```python
# 1. ライブラリのインポート
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# 2. データの読み込み
# パスの設定 
defected_data_path = r"../data/output/defect_data"
defected_image_path = r"../data/output/defect_data/defect_images" 
template_path = r"../data/input/template_matching_template"
defected_csv = "defects_data.csv"
defected_csv_path = os.path.join(defected_data_path, defected_csv)

# CSVファイル読み込み
print("=== データの読み込み ===")
df = pd.read_csv(defected_csv_path)
print(f"読み込んだデータ数: {len(df)}")

# work_predict_label=1のデータのみ抽出
df_filtered = df[df['work_predict_label'] == 1].copy()
print(f"処理対象データ数: {len(df_filtered)}")
```

```python
# 3. 前処理

def add_work_id(df):
    """
    データフレームにwork_id列を追加します
    
    引数:
    df (pandas.DataFrame): 処理対象のデータフレーム
    
    戻り値:
    pandas.DataFrame: work_id列が追加されたデータフレーム
    """
    # image_nameのユニークな値を取得してソート
    unique_images = sorted(df['image_name'].unique())
    
    # work_idの辞書を作成（2枚ずつ同じIDを割り当て）
    work_id_dict = {}
    for i in range(0, len(unique_images), 2):
        if i + 1 < len(unique_images):  # 2枚目が存在する場合
            work_id_dict[unique_images[i]] = i // 2
            work_id_dict[unique_images[i + 1]] = i // 2
        else:  # 最後の1枚の場合
            work_id_dict[unique_images[i]] = i // 2
    
    # work_id列を追加
    df['work_id'] = df['image_name'].map(work_id_dict)
    
    return df

def verify_image_paths(df, image_path):
    """
    画像ファイルの存在を確認します
    
    引数:
    df (pandas.DataFrame): 確認対象のデータフレーム
    image_path (str): 画像ファイルの基本パス
    
    戻り値:
    tuple: (存在する画像数, 存在しない画像数)
    """
    exists_count = 0
    not_exists_count = 0
    
    print("画像ファイルの確認中...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        full_path = os.path.join(image_path, row['defect_image_orig'])
        if os.path.exists(full_path):
            exists_count += 1
        else:
            not_exists_count += 1
            print(f"画像が存在しません: {full_path}")
    
    return exists_count, not_exists_count

def print_dataframe_info(df):
    """
    データフレームの基本情報を表示します
    
    引数:
    df (pandas.DataFrame): 表示対象のデータフレーム
    """
    print("\n=== データフレーム情報 ===")
    print(f"行数: {len(df)}")
    print(f"カラム: {df.columns.tolist()}")
    print("\nデータ型:")
    print(df.dtypes)
    print("\n欠損値の数:")
    print(df.isnull().sum())

# 前処理の実行
print("=== 前処理の開始 ===")

# work_id列の追加
df_filtered = add_work_id(df_filtered)

# 画像ファイルの存在確認
exists_count, not_exists_count = verify_image_paths(df_filtered, defected_image_path)
print(f"\n画像ファイル確認結果:")
print(f"存在する画像: {exists_count}")
print(f"存在しない画像: {not_exists_count}")

# データフレームの情報表示
print_dataframe_info(df_filtered)

# 作業用データフレームの保存（必要に応じて）
# df_filtered.to_csv('preprocessed_data.csv', index=False)
```

主な特徴：
1. 各関数に詳細なドキュメント文字列を追加
2. 処理を機能ごとに分割し、単一責任の原則に従う
3. エラーが発生しやすい箇所（ファイル存在確認など）を個別の関数に分離
4. 進捗状況の表示を追加（tqdmの使用）
5. データフレームの情報確認機能を追加

改良点：
- 関数の責任を明確に分離
- エラーハンドリングの改善
- 詳細なログ出力の追加
- コードの再利用性の向上
- 可読性の向上

続けて、テンプレートマッチングの実装部分を提供しましょうか？

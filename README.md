パターンマッチングによる分類システムのIPythonノートブックを作成します。まずはデータ読み込みの部分から始めましょう。

```markdown
# 欠陥検出システム - パターンマッチングによる分類

このノートブックでは、製品の欠陥検出をパターンマッチングで行います。特徴量による分類（work_predicted_label=1）で検出されたワークに対してのみ処理を行います。

## 1. 必要なライブラリのインポート
```

```python
import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
```

```markdown
## 2. データの読み込みと前処理
データセットの基本情報：
- NG（欠陥）データ：3件
- OK（非欠陥）データ：3827件
- 学習データとテストデータは同じものを使用（NGデータが少ないため）
```

```python
# データファイルのパスを設定
defected_data_path = r"../data/output/defect_data"
defected_image_path = r"../data/output/defect_data/defect_images"
defected_csv = "defects_data.csv"
defected_csv_path = os.path.join(defected_data_path, defected_csv)

# CSVファイルを読み込む
df = pd.read_csv(defected_csv_path)

# 特徴量による分類で欠陥ありと判定されたデータのみを抽出
df_filtered = df[df['work_predict_label'] == 1].copy()

# 基本情報の表示
print(f"元のデータ件数: {len(df)}")
print(f"work_predict_label=1のデータ件数: {len(df_filtered)}")
print(f"\nユニークなwork_id数: {df_filtered['work_id'].nunique()}")
print(f"実際の欠陥（defect_label=1）の数: {len(df_filtered[df_filtered['defect_label'] == 1])}")
```

```python
# 画像パスの確認と存在チェック
def verify_image_paths(df):
    """
    画像パスの存在を確認する関数
    """
    valid_paths = []
    invalid_paths = []
    
    for idx, row in df.iterrows():
        full_path = os.path.join(defected_data_path, row['defect_image_orig'])
        if os.path.exists(full_path):
            valid_paths.append(full_path)
        else:
            invalid_paths.append(full_path)
    
    print(f"有効な画像パス数: {len(valid_paths)}")
    print(f"無効な画像パス数: {len(invalid_paths)}")
    
    if invalid_paths:
        print("\n最初の数個の無効なパス:")
        for path in invalid_paths[:3]:
            print(path)
    
    return valid_paths, invalid_paths

# 画像パスの確認を実行
valid_paths, invalid_paths = verify_image_paths(df_filtered)
```

```markdown
## データの確認と前処理

上記のコードでは以下の処理を行っています：

1. 必要なライブラリのインポート
2. データファイルの読み込み
3. work_predict_label=1（特徴量による分類で欠陥ありと判定）のデータのフィルタリング
4. 画像パスの存在確認

次のステップでは：
1. 画像データの読み込みと前処理
2. パターンマッチングのためのテンプレート作成
3. パターンマッチングの実装

を行います。続けて進めますか？
```

この形式で進めていきたいと思いますが、いかがでしょうか？また、以下の点について確認させてください：

1. 画像ファイルの形式（.jpg, .png など）
2. 画像の典型的なサイズ
3. パターンマッチングで使用する特定のテンプレートの有無

これらの情報があれば、より適切なコードを提供できます。

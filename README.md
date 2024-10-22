パターンマッチングのための画像分類システムのipynbを生成します。まずはデータ読み込みの部分を作成します。

```markdown
# 欠陥検出システム - パターンマッチングによる分類
このノートブックでは、特徴量による分類で欠陥の可能性があると判断されたワークに対して、パターンマッチングによる詳細な分類を行います。

## 1. 必要なライブラリのインポート
```

```python
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
```

```markdown
## 2. データの読み込みと前処理
特徴量による分類（work_predict_label=1）のデータのみを処理します。
```

```python
# データファイルのパスを設定
defected_data_path = r"../data/output/defect_data"
defected_csv = "defects_data.csv"
defected_csv_path = os.path.join(defected_data_path, defected_csv)

# CSVファイルを読み込む
df = pd.read_csv(defected_csv_path)
print(f"元のデータフレームの形状：{df.shape}")

# work_predict_label=1のデータのみを抽出
df_filtered = df[df['work_predict_label'] == 1].copy()
print(f"フィルタリング後のデータフレームの形状：{df_filtered.shape}")

# 欠陥ラベルの分布を確認
print("\n欠陥ラベルの分布:")
print(df_filtered['defect_label'].value_counts())

# ワーク数の確認
n_works = df_filtered['work_id'].nunique()
print(f"\n総ワーク数: {n_works}")
```

```python
# 画像パスの作成と存在確認
def verify_image_path(row):
    img_path = os.path.join(defected_data_path, 'defect_images', row['defect_image_orig'])
    return os.path.exists(img_path)

# 画像パスの確認
df_filtered['image_exists'] = df_filtered.apply(verify_image_path, axis=1)
print("\n画像ファイルの存在確認:")
print(df_filtered['image_exists'].value_counts())

# サンプル画像の表示
def show_sample_images(df, num_samples=3):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    for i, (_, row) in enumerate(df.head(num_samples).iterrows()):
        img_path = os.path.join(defected_data_path, 'defect_images', row['defect_image_orig'])
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].set_title(f"Work ID: {row['work_id']}\nDefect: {row['defect_label']}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# サンプル画像の表示
show_sample_images(df_filtered)
```

```markdown
### データの基本情報:
- NGデータ（欠陥あり）: 3データ
- OKデータ（欠陥なし）: 03827データ
- 画像パス: defected_data_path + 'defect_images' + defect_image_orig

### 評価指標:
1. 欠陥ごとの評価:
   - 鋳巣検出率 = TP/(FN+TP) = 100%を目標
   - 鋳巣誤検出率 = FP/(TN+FP) を最小化

2. ワークごとの評価:
   - 見逃し率 = work_FN/(work_FN+workTP)
   - 見過ぎ率 = work_FP/(work_FP+workTN)

注意: ワークに一つでも欠陥があれば、そのワークは欠陥ありと判断します。
```

このコードでは以下の処理を行っています：
1. 必要なライブラリのインポート
2. CSVデータの読み込み
3. work_predict_label=1のデータのフィルタリング
4. 基本的なデータ確認（形状、分布など）
5. 画像パスの検証
6. サンプル画像の表示

次のステップとして、パターンマッチングのアルゴリズムの実装に進みたいと思いますが、その前に何か確認したい点はありますか？

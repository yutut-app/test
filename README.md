# 3. 前処理

本セクションでは、欠陥検出結果のデータに対して分類処理のための前処理を実施する。主な目的は、ワーク単位での分析を可能にするためのデータ構造の整備である。

## 関数の詳細説明

### add_work_id()
各画像に対してワークIDを付与する関数である：

1. 処理内容
   - 1つのワークに対して左右2枚の画像が存在
   - 画像名を基準に同一ワークの画像を特定
   - 連番のwork_idを付与

2. 処理手順
   - image_nameのユニーク値を取得・ソート
   - 2枚単位でwork_idを割り当て
   - 末尾が1枚のみの場合も考慮

3. データ構造
   ```
   例：
   image_name     work_id
   work1_left.jpg    0
   work1_right.jpg   0
   work2_left.jpg    1
   work2_right.jpg   1
   ```

### print_dataframe_info()
データフレームの状態を確認するための関数である：

1. 表示内容
   - データ行数
   - カラム一覧
   - 各カラムのデータ型
   - 欠損値の数

2. 確認ポイント
   - データの過不足
   - データ型の整合性
   - 欠損値の有無

## 前処理の実行フロー

1. work_idの付与
   - 画像名を基準にワークを特定
   - 同一ワークの画像に同じIDを付与

2. データ確認
   - 処理結果の妥当性確認
   - データの整合性チェック

3. データの保存（オプション）
   - 必要に応じて中間データとして保存
   - 後続の分析で再利用可能

この前処理により、後続の分析でワーク単位での評価や集計が可能になる。特に、1つのワークに対して左右の画像が存在する場合の整合性を保つために重要な処理となる。

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
df = add_work_id(df)

# データフレームの情報表示
print_dataframe_info(df)

# 作業用データフレームの保存（必要に応じて）
# df.to_csv('preprocessed_data.csv', index=False)

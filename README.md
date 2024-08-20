# 必要なライブラリをインポートします
import pandas as pd  # データ処理のためのライブラリ
import numpy as np   # 数値計算のためのライブラリ

# markdownセル
"""
# 1. データクリーニング：データの理解と前処理

データ分析の最初のステップは、データを理解し、きれいにすることです。これは「データクリーニング」と呼ばれます。
このステップでは、以下のことを行います：
1. データの中身を確認する
2. 不要なデータを削除する
3. データの形式を統一する
4. 欠損値（データが抜けている部分）を適切に処理する

これらの作業を行うことで、後の分析がスムーズに進められるようになります。
"""

# markdownセル
"""
## データ型ごとの列の一覧表示関数の定義

まず、データの中にどんな種類の情報があるかを確認するための関数を作ります。
この関数は、データの中の列（項目）を種類ごとに分けて表示してくれます。

データの種類には主に以下のようなものがあります：
- object: 文字列や混合データ
- int64: 整数
- float64: 小数点を含む数値
- datetime64: 日付と時刻

この関数を使うことで、どの列がどの種類のデータなのかが一目でわかります。
"""

# コードセル
# データ型ごとの列の一覧を表示する関数
def display_columns_types(df):
    print("\nデータ型ごとの列の一覧")
    data_types = df.dtypes.value_counts()  # 各データ型の数をカウント
    for dtype in data_types.index:
        print(f"\n{dtype}型の列:")
        # 特定のデータ型の列名をリストとして表示
        print(df.select_dtypes(include=[dtype]).columns.tolist())
    # データフレーム全体の行数と列数を表示
    print(f"\nデータフレームの形状: {df.shape}")

# markdownセル
"""
## データの前処理

次に、データを分析しやすい形に整えていきます。
不要なデータを削除したり、データの形式を統一したりします。

以下の手順で前処理を行います：
1. 全ての値が欠損している列を削除
2. 全ての値が同じ列を削除（これらの列は分析に役立たないため）
3. 特定の列の名前を変更
4. 欠損値を適切な値で埋める
5. 日付と時刻のデータを適切な形式に変換
6. 特定の列のデータ型を変換

これらの処理により、データがより扱いやすくなり、後の分析が正確になります。
"""

# コードセル
# データを読み込む（ここではdfという変数にデータが既に読み込まれていると仮定します）

# データ型ごとの列の一覧を表示（前処理前）
print("前処理前のデータ構造:")
display_columns_types(df)

# 元のデータフレーム列を保存（後で削除された列を確認するため）
original_columns = set(df.columns)

# 全ての値が欠損している列を削除
df = df.dropna(axis=1, how='all')
print("\n全ての値が欠損している列を削除しました。")

# 全ての値が同じ列を削除（比較できないため）
df = df.loc[:, df.round(2).nunique() != 1]
print("\n全ての値が同じ列を削除しました。")

# 削除した列を特定
removed_columns = original_columns - set(df.columns)

# 削除した列を表示
print("\n削除した列:")
for col in removed_columns:
    print(f"- {col}")

# 最初の列名を'image_position'に変更
df = df.rename(columns={df.columns[0]: 'image_position'})
print("\n最初の列名を'image_position'に変更しました。")

# 'image_position'列の欠損値を'O'で埋める
df['image_position'] = df['image_position'].fillna('O')
print("\n'image_position'列の欠損値を'O'で埋めました。")

# "低速変度数(VPC)"の列を数値型に変換
if "低速変度数(VPC)" in df.columns:
    df["低速変度数(VPC)"] = pd.to_numeric(df["低速変度数(VPC)"], errors='coerce')
    print("\n'低速変度数(VPC)'列を数値型に変換しました。")

# 日付と時刻の列を日時型に変換し、新しい日時列を追加
date_columns = ['日付', '出荷検査日付', '加工検査日付']
time_columns = ['時刻', '出荷検査時刻', '加工検査時刻']
datetime_columns = ['日時', '出荷検査日時', '加工検査日時']

for date_col, time_col, datetime_col in zip(date_columns, time_columns, datetime_columns):
    # 日付と時刻を文字列として結合
    datetime_str = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
    # 日時型に変換
    df[datetime_col] = pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M:%S', errors='coerce')
print("\n日付と時刻の列を日時型に変換し、新しい日時列を追加しました。")

# 特定の列を数値部分のみの文字列型に変換
id_columns = ['工程内検査1コード', '工程内検査2コード', '加工先1コード', '加工先2コード']
for col in id_columns:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) else None)
print("\n特定の列を数値部分のみの文字列型に変換しました。")

# データ型ごとの列の一覧を表示（前処理後）
print("\n前処理後のデータ構造:")
display_columns_types(df)

# markdownセル
"""
## 数値データの基本統計量の確認

数値で表されているデータについて、平均値や最大値、最小値などの基本的な統計情報を確認します。
これにより、データの全体的な特徴を把握できます。

確認する統計量：
- count: データの数
- mean: 平均値
- std: 標準偏差（データのばらつきを示す）
- min: 最小値
- 25%: 第1四分位数（データを小さい順に並べたときの25%位置の値）
- 50%: 中央値
- 75%: 第3四分位数（データを小さい順に並べたときの75%位置の値）
- max: 最大値

これらの値を見ることで、データの分布や特徴を理解できます。
例えば、平均値と中央値が大きく異なる場合、データに偏りがある可能性があります。
"""

# コードセル
# 数値データの基本統計量を表示
print("数値データの基本統計量:")
display(df.describe())

# markdownセル
"""
## カテゴリデータの確認

文字や記号で表されているデータ（カテゴリデータ）について、それぞれの値がどれくらいの頻度で出現するかを確認します。
これにより、データの分布や特徴を理解できます。

例えば、ある製品のカテゴリごとの生産数や、特定の状態の発生頻度などを確認できます。
頻度が極端に低い or 高いカテゴリがあれば、それが何を意味するのか考察することが重要です。
"""

# コードセル
# カテゴリデータの値ごとの出現回数を表示
categorical_columns = df.select_dtypes(include=[object]).columns
for col in categorical_columns:
    print(f"\n{col}の値ごとの出現回数:")
    display(df[col].value_counts())
    print(f"{col}のユニーク値の数: {df[col].nunique()}")

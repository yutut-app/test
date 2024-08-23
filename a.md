# 必要なライブラリをインポートします
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 日本語フォントの設定
# これにより、グラフ上の日本語が正しく表示されます
plt.rcParams['font.family'] = 'MS Gothic'  # または 'IPAexGothic', 'Yu Gothic'などを試してみてください

# データの読み込み
# CSVファイルからデータを読み込みます。これにより、データを分析できる形式で取得できます。
df = pd.read_csv('casting_data.csv')

# データの前処理
# 日付列を日付型に変換します。これにより、日付データを正しく扱えるようになります。
date_columns = ['日時', '出荷検査日時', '加工検査日時']
for col in date_columns:
    df[col] = pd.to_datetime(df[col])

# 時系列順にソート
# データを日時順に並べ替えます。これにより、時間の流れに沿った分析が可能になります。
df = df.sort_values('日時')

# 鋳造条件の列名を取得（int型とfloat型の列）
# 目的変数を除いた数値型の列を鋳造条件として扱います。
casting_condition_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
casting_condition_columns = [col for col in casting_condition_columns if col != '目的変数']

# 出力ディレクトリの作成
# グラフを保存するためのフォルダを作成します。
output_dir = r'..\data\output\eda\鋳造機と鋳造条件の関係'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# 各鋳造条件ごとに散布図を作成
for condition in casting_condition_columns:
    # 新しい図を作成
    plt.figure(figsize=(12, 8))
    
    # 散布図をプロット
    # 目的変数が0（OK）の場合は青、1（NG）の場合はオレンジで表示します
    for target in [0, 1]:
        mask = df['目的変数'] == target
        color = 'blue' if target == 0 else 'orange'
        label = 'OK' if target == 0 else 'NG'
        plt.scatter(df.loc[mask, condition], df.loc[mask, '鋳造機名'], 
                    c=color, label=label, alpha=0.6)
    
    # グラフの設定
    plt.title(f'{condition}と鋳造機名の関係')
    plt.xlabel(condition)
    plt.ylabel('鋳造機名')
    plt.legend()
    
    # グラフの保存
    filename = f'timevis_{condition}_{current_time}.pdf'
    plt.savefig(os.path.join(output_dir, filename))
    
    # メモリの解放
    plt.close()
    
    # 表示したい場合は以下のコメントを外してください
    # plt.show()

print("すべての散布図がPDFとして保存されました。")

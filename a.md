import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'  # または 'IPAexGothic', 'Yu Gothic'などを試してみてください

# データの読み込み
df = pd.read_csv('casting_data.csv')

# データの前処理
date_columns = ['日時', '出荷検査日時', '加工検査日時']
for col in date_columns:
    df[col] = pd.to_datetime(df[col])

# 時系列順にソート
df = df.sort_values('日時')

# 鋳造条件の列名を取得（int型とfloat型の列）
casting_condition_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
casting_condition_columns = [col for col in casting_condition_columns if col != '目的変数']

# 出力ディレクトリの作成
output_dir = r'..\data\output\time_vis'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# 各鋳造条件ごとに散布図を作成してPDF保存
for condition in casting_condition_columns:
    plt.figure(figsize=(12, 8))
    
    # 目的変数が0（OK）のデータ
    ok_data = df[df['目的変数'] == 0]
    plt.scatter(ok_data[condition], ok_data['鋳造機名'], color='blue', label='OK', alpha=0.5)
    
    # 目的変数が1（NG）のデータ
    ng_data = df[df['目的変数'] == 1]
    plt.scatter(ng_data[condition], ng_data['鋳造機名'], color='orange', label='NG', alpha=0.5)
    
    plt.xlabel(condition)
    plt.ylabel('鋳造機名')
    plt.title(f'{condition}の散布図')
    plt.legend()
    
    # グラフの保存
    filename = f'timevis_{condition}_{current_time}.pdf'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    # plt.show()  # 可視化する際はこのコメントアウトを外してください

print("全ての散布図のPDFが保存されました。")

# 必要なライブラリをインポート
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'
# または 'IPAexGothic', 'Yu Gothic'などを試してみてください

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
output_dir = r'..\data\output\eda\鋳造機と鋳造条件の関係'
os.makedirs(output_dir, exist_ok=True)

# 各鋳造条件に対してcatplotを作成し、PDFとして保存
for condition in casting_condition_columns:
    plt.figure(figsize=(12, 8))
    sns.catplot(data=df, x=condition, y='鋳造機名', hue='目的変数', kind='strip', 
                palette={0: 'blue', 1: 'orange'}, height=6, aspect=2)
    
    plt.title(f'鋳造機名と{condition}の関係')
    plt.tight_layout()
    
    # 現在の日時を取得してファイル名に使用
    current_time = datetime.now().strftime("%y%m%d%H%M")
    filename = f'timevis_{condition}_{current_time}.pdf'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    # plt.show()  # 可視化する際はこのコメントを外してください

print("全ての図表がPDFとして保存されました。")

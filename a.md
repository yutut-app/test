import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import numpy as np

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
output_dir = r'..\data\output\eda'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_鋳造機と鋳造条件の関係_{current_time}.pdf')

from matplotlib.backends.backend_pdf import PdfPages

# jitterを追加する関数
def add_jitter(values, jitter_amount=0.3):
    return values + np.random.uniform(-jitter_amount, jitter_amount, len(values))

# PDFファイルを開く
with PdfPages(pdf_filename) as pdf:
    # 各鋳造条件に対してプロットを作成
    for condition in casting_condition_columns:
        # プロットの作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # OKとNGのデータを分離
        df_ok = df[df['目的変数'] == 0]
        df_ng = df[df['目的変数'] == 1]
        
        # OKのデータをプロット（透明度を下げる）
        sns.stripplot(data=df_ok, x=condition, y='鋳造機名', color='blue', alpha=0.3, 
                      jitter=True, size=5, ax=ax)
        
        # NGのデータをプロット（前面に、大きく、透明度を上げる）
        sns.stripplot(data=df_ng, x=condition, y='鋳造機名', color='red', alpha=1.0, 
                      jitter=True, size=10, ax=ax)
        
        # タイトルと軸ラベルの設定
        plt.title(f'鋳造機名と{condition}の関係')
        plt.xlabel(condition)
        plt.ylabel('鋳造機名')
        
        # 凡例の設定
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='OK (0)',
                                  markerfacecolor='blue', markersize=10, alpha=0.3),
                           Line2D([0], [0], marker='o', color='w', label='NG (1)',
                                  markerfacecolor='red', markersize=15)]
        ax.legend(handles=legend_elements, title='目的変数')
        
        # グラフの調整
        plt.tight_layout()
        
        # PDFに追加
        pdf.savefig(fig)
        
        # プロットをクローズ
        plt.close(fig)

print(f"グラフをPDFに保存しました: {pdf_filename}")

# 可視化する際は以下のコメントを外してください
# plt.show()

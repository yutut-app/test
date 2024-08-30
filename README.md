import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

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

# 時間帯を抽出
df['時間帯'] = df['日時'].dt.hour

# NG率の計算
def calculate_ng_rate(group):
    total = len(group)
    ng_count = (group == 1).sum()
    return ng_count / total * 100 if total > 0 else 0

# 出力ディレクトリの作成
output_dir = r'..\data\output\eda\NG数の時系列の偏り'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# グラフを表示するかどうかのフラグ
show_plots = False  # Trueにするとグラフを表示、Falseにすると表示しない

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_時間の偏り_全鋳造機名_{current_time}.pdf')

with PdfPages(pdf_filename) as pdf:
    for machine in df['鋳造機名'].unique():
        # 鋳造機名ごとのデータをフィルタリング
        df_machine = df[df['鋳造機名'] == machine]
        
        # 品番の数を取得
        product_count = df_machine['品番'].nunique()
        
        # サブプロットの行数と列数を決定
        rows = (product_count + 1) // 2  # 切り上げ除算
        cols = 2 if product_count > 1 else 1

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows), squeeze=False)
        fig.suptitle(f'{machine}の時間帯ごとのNG率', fontsize=16)

        for i, product in enumerate(df_machine['品番'].unique()):
            ax = axes[i//2, i%2]
            
            # 品番ごとのデータをフィルタリング
            df_product = df_machine[df_machine['品番'] == product]
            
            # 時間帯ごとのNG率を計算
            ng_rate = df_product.groupby('時間帯')['目的変数'].apply(calculate_ng_rate).reset_index()
            ng_rate.columns = ['時間帯', 'NG率']
            
            # プロット
            ax.plot(ng_rate['時間帯'], ng_rate['NG率'], marker='o')
            ax.set_title(f'品番: {product}')
            ax.set_xlabel('時間帯')
            ax.set_ylabel('NG率 [%]')
            ax.set_xticks(range(0, 24))
            ax.set_ylim(0, 100)
            
        # 余分なサブプロットを削除
        for i in range(product_count, rows*cols):
            fig.delaxes(axes[i//2, i%2])

        plt.tight_layout()
        pdf.savefig(fig)
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

print(f"グラフをPDFに保存しました: {pdf_filename}")

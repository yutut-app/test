import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import os
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# データの読み込み
df = pd.read_csv('casting_process_data.csv')

# データの前処理
date_columns = ['日時', '出荷検査日時', '加工検査日時']
for col in date_columns:
    df[col] = pd.to_datetime(df[col])

# 出力ディレクトリの作成
output_dir = r'..\data\output\time_vis'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得してファイル名に使用
current_time = datetime.now().strftime("%y%m%d%H%M")

# 鋳造条件のカラムを取得（目的変数を除く数値列）
casting_conditions = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
casting_conditions.remove('目的変数')

# PDFファイルの作成
pdf_filename = os.path.join(output_dir, f'time_vis_鋳造条件_{current_time}.pdf')
with PdfPages(pdf_filename) as pdf:
    for condition in casting_conditions:
        plt.figure(figsize=(15, 8))
        sns.set_style("whitegrid")

        # OKとNGのデータを分けて、それぞれ異なる色で表示
        ok_data = df[df['目的変数'] == 0]
        ng_data = df[df['目的変数'] == 1]

        plt.plot(ok_data['日時'], ok_data[condition], color='blue', label='OK', marker='o')
        plt.plot(ng_data['日時'], ng_data[condition], color='red', label='NG', marker='x')

        # 5分以上の間隔があるデータポイントの間は線を繋げない
        for i in range(1, len(df)):
            if (df['日時'].iloc[i] - df['日時'].iloc[i-1]) > timedelta(minutes=5):
                plt.axvline(x=df['日時'].iloc[i], color='gray', linestyle='--', alpha=0.5)

        plt.title(f'{condition}の時系列変化', fontsize=16)
        plt.xlabel('日時', fontsize=12)
        plt.ylabel(f'{condition}の値', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # グラフをPDFに保存
        pdf.savefig()
        plt.close()

        # 鋳造機名ごとの分析
        plt.figure(figsize=(15, 8))
        for machine in df['鋳造機名'].unique():
            machine_data = df[df['鋳造機名'] == machine]
            plt.plot(machine_data['日時'], machine_data[condition], label=machine)

        plt.title(f'鋳造機名ごとの{condition}の時系列変化', fontsize=16)
        plt.xlabel('日時', fontsize=12)
        plt.ylabel(f'{condition}の値', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # グラフをPDFに保存
        pdf.savefig()
        plt.close()

        # 品番（型番）ごとの分析
        plt.figure(figsize=(15, 8))
        for product in df['品番'].unique():
            product_data = df[df['品番'] == product]
            plt.plot(product_data['日時'], product_data[condition], label=product)

        plt.title(f'品番ごとの{condition}の時系列変化', fontsize=16)
        plt.xlabel('日時', fontsize=12)
        plt.ylabel(f'{condition}の値', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # グラフをPDFに保存
        pdf.savefig()
        plt.close()

# OKとNGの割合の時系列変化
df['date'] = df['日時'].dt.date
daily_ratio = df.groupby('date')['目的変数'].mean()

plt.figure(figsize=(15, 8))
plt.plot(daily_ratio.index, daily_ratio.values, marker='o')
plt.title('日ごとのNG率の推移', fontsize=16)
plt.xlabel('日付', fontsize=12)
plt.ylabel('NG率', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# グラフをPDFに保存
pdf.savefig()
plt.close()

print(f"グラフがPDFファイルとして保存されました: {pdf_filename}")

# plt.show()をコメントアウト
# plt.show()

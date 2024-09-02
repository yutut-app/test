import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'  # または 'IPAexGothic', 'Yu Gothic'などを試してみてください

# データの読み込み
df = pd.read_csv('casting_data.csv')

# データの前処理
df['日時'] = pd.to_datetime(df['日時'])
df['週'] = (df['日時'] - df['日時'].min()).dt.days // 7 + 1

# 出力ディレクトリの作成
output_dir = r'..\data\output\eda\NG数の時系列の偏り\週ごとの偏り'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# グラフを表示するかどうかのフラグ
show_plots = False  # Trueにするとグラフを表示、Falseにすると表示しない

# NG率の計算関数
def calculate_ng_rate(group):
    total = len(group)
    ng_count = (group['目的変数'] == 1).sum()
    return ng_count / total * 100 if total > 0 else 0, ng_count, total

# 週ごとのデータ数を計算する関数
def count_days_in_week(group):
    return group['日時'].dt.date.nunique()

# 稼働時間を計算する関数
def get_operation_hours(group):
    start_time = group['日時'].min().strftime('%H:%M')
    end_time = group['日時'].max().strftime('%H:%M')
    return f"{start_time}~{end_time}"

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_週ごとの偏り_全鋳造機_{current_time}.pdf')

with PdfPages(pdf_filename) as pdf:
    # 鋳造機名ごとにプロットを作成
    for machine in df['鋳造機名'].unique():
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 鋳造機名でフィルタリング
        df_machine = df[df['鋳造機名'] == machine]
        
        # 品番ごとにNG率を計算し、プロット
        for product in df_machine['品番'].unique():
            df_product = df_machine[df_machine['品番'] == product]
            ng_rates = df_product.groupby('週').apply(calculate_ng_rate)
            
            x = ng_rates.index
            y = [rate for rate, _, _ in ng_rates.values]
            
            ax.plot(x, y, label=f'品番 {product}', marker='o')
            
            # NG率が7.5%以上の場合、NG数を表示
            for i, (rate, ng_count, total) in enumerate(ng_rates.values):
                if rate >= 7.5:
                    ax.text(x[i], y[i], f'{ng_count}/{total}', ha='center', va='bottom')
        
        # 週ごとのデータ数を計算
        days_in_week = df_machine.groupby('週').apply(count_days_in_week)
        
        ax.set_xlabel('週')
        ax.set_ylabel('NG率 [%]')
        ax.set_title(f'{machine}の週別NG率')
        ax.set_xticks(range(1, df_machine['週'].max() + 1))
        ax.set_ylim(0, 100)
        plt.grid(True)
        
        # 7日未満の週にテキストを追加
        for week, days in days_in_week.items():
            if days < 7:
                ax.text(week, ax.get_ylim()[1], f'({days})', ha='center', va='bottom')
        
        # 稼働時間を含む凡例を作成
        legend_elements = []
        for product in df_machine['品番'].unique():
            df_product = df_machine[df_machine['品番'] == product]
            operation_hours = df_product.groupby('週').apply(get_operation_hours).mode().values[0]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color=ax.get_lines()[-1].get_color(), 
                                              label=f'品番 {product} (稼働時間: {operation_hours})'))
        
        ax.legend(handles=legend_elements)
        
        # PDFに追加
        pdf.savefig(fig)
        
        # PNGとして保存
        png_filename = os.path.join(output_dir, f'vis_週ごとの偏り_{machine}_{current_time}.png')
        plt.savefig(png_filename)
        
        # グラフを表示（フラグがTrueの場合）
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    print(f"全鋳造機のグラフをPDFに保存しました: {pdf_filename}")

print("各鋳造機のグラフをPNGに保存しました。")

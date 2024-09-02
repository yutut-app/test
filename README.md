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
df['日時'] = pd.to_datetime(df['日時'])
df['時間'] = df['日時'].dt.hour

# 出力ディレクトリの作成
output_dir = r'..\data\output\eda\NG数の時系列の偏り\時間の偏り'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# グラフを表示するかどうかのフラグ
show_plots = False  # Trueにするとグラフを表示、Falseにすると表示しない

# NG率の計算関数
def calculate_ng_rate(group):
    total = group.shape[0]
    if total == 0:
        return None  # データが無い場合はNoneを返す
    ng_count = group[group['目的変数'] == 1].shape[0]
    return (ng_count, total, (ng_count / total) * 100 if total > 0 else None)

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_時間の偏り_全鋳造機_{current_time}.pdf')

with PdfPages(pdf_filename) as pdf:
    # 鋳造機名ごとにプロットを作成
    for machine in df['鋳造機名'].unique():
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 鋳造機名でフィルタリング
        df_machine = df[df['鋳造機名'] == machine]
        
        # 品番ごとにNG率を計算し、プロット
        for product in df_machine['品番'].unique():
            df_product = df_machine[df_machine['品番'] == product]
            ng_rates = df_product.groupby('時間').apply(calculate_ng_rate)
            
            # Noneの値（データ無し）を除外してプロット
            valid_data = ng_rates.dropna()
            x_values = [i[0] for i in valid_data.index]
            y_values = [i[2] for i in valid_data.values]
            
            line, = ax.plot(x_values, y_values, label=f'品番 {product}', marker='o')
            
            # NG率が7.5%以上の点にテキストを追加
            for x, y, (ng, total, _) in zip(x_values, y_values, valid_data.values):
                if y >= 7.5:
                    ax.annotate(f"{ng}/{total}", (x, y), xytext=(0, 10), 
                                textcoords='offset points', ha='center', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                                fontsize=12)  # フォントサイズを12に設定
        
        ax.set_xlabel('時間 (時)', fontsize=14)
        ax.set_ylabel('NG率 [%]', fontsize=14)
        ax.set_title(f'{machine}の時間帯別NG率', fontsize=16)
        ax.set_xticks(range(0, 24))
        ax.set_xticklabels(range(0, 24), fontsize=12)
        ax.set_yticklabels(ax.get_yticks(), fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=12)
        plt.grid(True)
        
        # PDFに追加
        pdf.savefig(fig)
        
        # PNGとして保存
        png_filename = os.path.join(output_dir, f'vis_時間の偏り_{machine}_{current_time}.png')
        plt.savefig(png_filename)
        
        # グラフを表示（フラグがTrueの場合）
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    print(f"全鋳造機のグラフをPDFに保存しました: {pdf_filename}")

print("各鋳造機のグラフをPNGに保存しました。")

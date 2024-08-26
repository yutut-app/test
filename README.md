
# NG率の計算
def calculate_ng_rate(data):
    total = len(data)
    ng_count = (data == 1).sum()
    return (ng_count / total) * 100 if total > 0 else 0

# 出力ディレクトリの作成
output_dir = r'..\data\output\eda'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

from matplotlib.backends.backend_pdf import PdfPages

# グラフを表示するかどうかのフラグ
show_plots = True  # Trueにするとグラフを表示、Falseにすると表示しない

# 鋳造機名ごとにプロットを作成
for machine in df['鋳造機名'].unique():
    # 鋳造機名ごとのデータをフィルタリング
    df_machine = df[df['鋳造機名'] == machine]
    
    # 品番ごとのNG率を計算
    ng_rate_by_product = df_machine.groupby('品番')['目的変数'].agg(calculate_ng_rate).reset_index()
    ng_rate_by_product.columns = ['品番', 'NG率']
    
    # 鋳造機名ごとの総データ数を計算
    total_count = len(df_machine)
    
    # PDFファイルを作成
    pdf_filename = os.path.join(output_dir, f'vis_{machine}の品番ごとNG率_{current_time}.pdf')
    
    with PdfPages(pdf_filename) as pdf:
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ng_rate_by_product.plot(x='品番', y='NG率', kind='bar', ax=ax)
        plt.title(f'{machine}の品番ごとの渦流探傷NG率 (n={total_count})')
        plt.xlabel('品番')
        plt.ylabel('NG率 [%]')
        plt.ylim(0, 100)  # Y軸の最大値を100%に設定

        # 各棒グラフの上に値とテキストを表示
        for i, bar in enumerate(bars.patches):
            height = bar.get_height()
            product = ng_rate_by_product['品番'].iloc[i]
            ng_count = df_machine[(df_machine['品番'] == product) & (df_machine['目的変数'] == 1)].shape[0]
            total_count = df_machine[df_machine['品番'] == product].shape[0]
            text = f"{height:.2f}% ({ng_count}/{total_count})"
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    text,
                    ha='center', va='bottom')

        plt.tight_layout()
        pdf.savefig(fig)
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    print(f"{machine}のグラフをPDFに保存しました: {pdf_filename}")

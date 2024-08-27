# NG率の計算
def calculate_ng_rate(group):
    total = len(group)
    ng_count = (group == 1).sum()
    return (ng_count, total)

# 出力ディレクトリの作成
output_dir = r'..\data\output\eda'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

from matplotlib.backends.backend_pdf import PdfPages

# グラフを表示するかどうかのフラグ
show_plots = True  # Trueにするとグラフを表示、Falseにすると表示しない

# 品番ごとのNG率を計算
ng_rate_by_product = df.groupby(['鋳造機名', '品番'])['目的変数'].apply(calculate_ng_rate).reset_index()
ng_rate_by_product.columns = ['鋳造機名', '品番', 'NG_count_total']
ng_rate_by_product[['NG_count', 'Total_count']] = pd.DataFrame(ng_rate_by_product['NG_count_total'].tolist(), index=ng_rate_by_product.index)
ng_rate_by_product['NG率'] = ng_rate_by_product['NG_count'] / ng_rate_by_product['Total_count'] * 100

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_鋳造機名ごとの品番ごとNG率_{current_time}.pdf')

with PdfPages(pdf_filename) as pdf:
    fig, ax = plt.subplots(figsize=(15, 8))
    
    bar_width = 0.35
    index = np.arange(len(ng_rate_by_product['品番'].unique()))
    
    for i, machine in enumerate(df['鋳造機名'].unique()):
        data = ng_rate_by_product[ng_rate_by_product['鋳造機名'] == machine]
        bars = ax.bar(index + i*bar_width, data['NG率'], bar_width, label=machine)
        
        # 各棒グラフの上に値とテキストを表示
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ng_count = data['NG_count'].iloc[j]
            total_count = data['Total_count'].iloc[j]
            if total_count == 0:
                text = "0.00%(0/0)"
            else:
                text = f"{height:.2f}% ({ng_count}/{total_count})"
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    text,
                    ha='center', va='bottom', rotation=90)

    plt.title(f'鋳造機名ごとの品番ごとの渦流探傷NG率')
    plt.xlabel('品番')
    plt.ylabel('NG率 [%]')
    plt.ylim(0, 100)  # Y軸の最大値を100%に設定
    plt.xticks(index + bar_width/2, ng_rate_by_product['品番'].unique())
    plt.legend()
    plt.tight_layout()
    
    pdf.savefig(fig)
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

print(f"グラフをPDFに保存しました: {pdf_filename}")

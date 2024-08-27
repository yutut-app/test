# NG率の計算
def calculate_ng_rate(data):
    total = len(data)
    ng_count = (data == 1).sum()
    return (ng_count / total) * 100 if total > 0 else 0

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

from matplotlib.backends.backend_pdf import PdfPages

# グラフを表示するかどうかのフラグ
show_plots = True  # Trueにするとグラフを表示、Falseにすると表示しない

# 全鋳造機名のデータを使用してNG率を計算
ng_rate_by_product_machine = df.groupby(['品番', '鋳造機名'])['目的変数'].agg(calculate_ng_rate).reset_index()
ng_rate_by_product_machine.columns = ['品番', '鋳造機名', 'NG率']

# 総データ数を計算
total_count = len(df)

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_全鋳造機名の品番ごとNG率_{current_time}.pdf')

with PdfPages(pdf_filename) as pdf:
    fig, ax = plt.subplots(figsize=(15, 8))
    bars = sns.barplot(x='品番', y='NG率', hue='鋳造機名', data=ng_rate_by_product_machine, ax=ax)
    plt.title(f'全鋳造機名の品番ごとの渦流探傷NG率 (n={total_count})')
    plt.xlabel('品番')
    plt.ylabel('NG率 [%]')
    plt.ylim(0, 100)  # Y軸の最大値を100%に設定
    
    # 各棒グラフの上に値とテキストを表示
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        product = ng_rate_by_product_machine['品番'].iloc[i // len(df['鋳造機名'].unique())]
        machine = ng_rate_by_product_machine['鋳造機名'].iloc[i % len(df['鋳造機名'].unique())]
        ng_count = df[(df['品番'] == product) & (df['鋳造機名'] == machine) & (df['目的変数'] == 1)].shape[0]
        total_count = df[(df['品番'] == product) & (df['鋳造機名'] == machine)].shape[0]
        text = f"{height:.2f}%\n({ng_count}/{total_count})"
        ax.text(bar.get_x() + bar.get_width()/2., height,
                text,
                ha='center', va='bottom', fontsize=8)
    
    plt.xticks(rotation=45)
    plt.legend(title='鋳造機名')
    plt.tight_layout()
    pdf.savefig(fig)
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

print(f"全鋳造機名のグラフをPDFに保存しました: {pdf_filename}")

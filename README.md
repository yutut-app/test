# NG率の計算
def calculate_ng_rate(group):
    total = len(group)
    ng_count = (group['目的変数'] == 1).sum()
    return (ng_count / total) * 100 if total > 0 else 0

# 品番と鋳造機名ごとのNG率を計算
ng_rate = df.groupby(['品番', '鋳造機名']).apply(calculate_ng_rate).reset_index()
ng_rate.columns = ['品番', '鋳造機名', 'NG率']


# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_品番ごとの鋳造機別NG率_{current_time}.pdf')

from matplotlib.backends.backend_pdf import PdfPages

# グラフを表示するかどうかのフラグ
show_plots = True  # Trueにするとグラフを表示、Falseにすると表示しない

# 棒グラフ上に値を表示する関数
def add_value_labels(ax, spacing=5):
    for rect in ax.patches:
        height = rect.get_height()
        if height > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + spacing,
                f'{height:.2f}%\n({int(rect.get_width())})',
                ha='center', va='bottom', rotation=0
            )

# PDFファイルを開く
with PdfPages(pdf_filename) as pdf:
    # 鋳造機名ごとのNG率プロット
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 棒グラフの作成
    sns.barplot(x='品番', y='NG率', hue='鋳造機名', data=ng_rate, ax=ax)
    
    # n数の計算
    n = len(df)
    
    plt.title(f'品番ごとの鋳造機別渦流探傷NG率 (n={n})')
    plt.xlabel('品番')
    plt.ylabel('NG率 [%]')
    plt.ylim(0, 100)  # Y軸の最大値を100%に設定
    
    # 値のラベルを追加
    add_value_labels(ax)
    
    plt.legend(title='鋳造機名')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    pdf.savefig(fig)
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

print(f"グラフをPDFに保存しました: {pdf_filename}")

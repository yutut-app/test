# NG率の計算
def calculate_ng_rate(group):
    total = len(group)
    ng_count = (group['目的変数'] == 1).sum()
    return (ng_count / total) * 100 if total > 0 else 0

# 鋳造機名と品番ごとのNG率を計算
ng_rate_by_machine = df.groupby('鋳造機名').apply(calculate_ng_rate).reset_index()
ng_rate_by_machine.columns = ['鋳造機名', 'NG率']

ng_rate_by_product = df.groupby('品番').apply(calculate_ng_rate).reset_index()
ng_rate_by_product.columns = ['品番', 'NG率']

# 出力ディレクトリの作成
output_dir = r'..\data\output\eda'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_各鋳造条件におけるNG率_{current_time}.pdf')

# グラフを表示するかどうかのフラグ
show_plots = True  # Trueにするとグラフを表示、Falseにすると表示しない

# PDFファイルを開く
with PdfPages(pdf_filename) as pdf:
    # 鋳造機名ごとのNG率プロット
    fig, ax = plt.subplots(figsize=(12, 6))
    ng_rate_by_machine.plot(x='鋳造機名', y='NG率', kind='bar', ax=ax)
    plt.title('鋳造機名ごとの渦流探傷NG率')
    plt.xlabel('鋳造機名')
    plt.ylabel('NG率 [%]')
    plt.ylim(0, 100)  # Y軸の最大値を100%に設定
    plt.tight_layout()
    pdf.savefig(fig)
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # 品番ごとのNG率プロット
    fig, ax = plt.subplots(figsize=(12, 6))
    ng_rate_by_product.plot(x='品番', y='NG率', kind='bar', ax=ax)
    plt.title('品番ごとの渦流探傷NG率')
    plt.xlabel('品番')
    plt.ylabel('NG率 [%]')
    plt.ylim(0, 100)  # Y軸の最大値を100%に設定
    plt.tight_layout()
    pdf.savefig(fig)
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

print(f"グラフをPDFに保存しました: {pdf_filename}")

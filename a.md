# 出力ディレクトリの作成
output_dir = r'..\data\output\eda'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_各鋳造条件におけるNG率_{current_time}.pdf')

# グラフを表示するかどうかのフラグ
show_plots = True  # Trueにするとグラフを表示、Falseにすると表示しない

# NG率を計算する関数
def calculate_ng_rate(group):
    total = len(group)
    ng_count = group['目的変数'].sum()
    return ng_count / total if total > 0 else 0

# PDFファイルを開く
with PdfPages(pdf_filename) as pdf:
    # 各鋳造条件に対してプロットを作成
    for condition in casting_condition_columns:
        # 鋳造条件を10分位に分割
        df['condition_bin'] = pd.qcut(df[condition], q=10, labels=False, duplicates='drop')
        
        # 各ビンごとのNG率を計算
        ng_rates = df.groupby('condition_bin').apply(calculate_ng_rate).reset_index()
        ng_rates.columns = ['condition_bin', 'ng_rate']
        
        # ビンの中央値を計算
        bin_means = df.groupby('condition_bin')[condition].mean().reset_index()
        
        # NG率とビンの中央値をマージ
        plot_data = pd.merge(ng_rates, bin_means, on='condition_bin')
        
        # プロットの作成
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 棒グラフの作成
        plot_data.plot(kind='bar', x=condition, y='ng_rate', ax=ax)
        
        # タイトルと軸ラベルの設定
        plt.title(f'{condition}における渦流探傷のNG率')
        plt.xlabel(condition)
        plt.ylabel('NG率')
        
        # x軸のラベルを回転
        plt.xticks(rotation=45, ha='right')
        
        # グラフの調整
        plt.tight_layout()
        
        # PDFに追加
        pdf.savefig(fig)
        
        # グラフを表示（フラグがTrueの場合）
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

print(f"グラフをPDFに保存しました: {pdf_filename}")

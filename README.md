# NG率の計算
def calculate_ng_rate(group):
    total = len(group)
    ng_count = (group == 1).sum()
    return (ng_count, total, ng_count / total * 100 if total > 0 else 0)

# 出力ディレクトリの作成
output_dir = r'..\data\output\eda\NG数の時系列の偏り\時間の偏り'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# グラフを表示するかどうかのフラグ
show_plots = False  # Trueにするとグラフを表示、Falseにすると表示しない

# 時間ごとのNG率を計算
df['時間'] = df['日時'].dt.hour
ng_rate_by_hour = df.groupby(['鋳造機名', '品番', '時間'])['目的変数'].apply(calculate_ng_rate).reset_index()
ng_rate_by_hour.columns = ['鋳造機名', '品番', '時間', 'NG_データ']
ng_rate_by_hour[['NG回数', '全回数', 'NG率']] = pd.DataFrame(ng_rate_by_hour['NG_データ'].tolist(), index=ng_rate_by_hour.index)
ng_rate_by_hour = ng_rate_by_hour.drop('NG_データ', axis=1)

# 色の設定
colors = plt.cm.rainbow(np.linspace(0, 1, len(df['品番'].unique())))

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_時間の偏り_全鋳造機名_{current_time}.pdf')

with PdfPages(pdf_filename) as pdf:
    # 鋳造機名ごとにプロットを作成
    for machine in df['鋳造機名'].unique():
        fig, ax = plt.subplots(figsize=(15, 8))
        
        data = ng_rate_by_hour[ng_rate_by_hour['鋳造機名'] == machine]
        
        for i, product in enumerate(df['品番'].unique()):
            product_data = data[data['品番'] == product]
            ax.plot(product_data['時間'], product_data['NG率'], marker='o', label=product, color=colors[i])
        
        ax.set_xlabel('時間 (時)')
        ax.set_ylabel('NG率 [%]')
        ax.set_title(f'{machine}の時間ごとNG率')
        ax.set_xticks(range(24))
        ax.set_xlim(0, 23)
        ax.set_ylim(0, 100)
        ax.legend()
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

    print(f"全鋳造機名のグラフをPDFに保存しました: {pdf_filename}")
    print(f"各鋳造機名のグラフをPNGに保存しました: {output_dir}")

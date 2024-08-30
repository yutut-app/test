# データの前処理
df['日時'] = pd.to_datetime(df['日時'])
df['週'] = df['日時'].dt.to_period('W')

# 出力ディレクトリの作成
output_dir = r'..\data\output\eda\NG数の時系列の偏り\週ごとの偏り'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# グラフを表示するかどうかのフラグ
show_plots = False  # Trueにするとグラフを表示、Falseにすると表示しない

# NG率の計算関数
def calculate_ng_rate(group):
    return group[group['目的変数'] == 1].shape[0] / group.shape[0] * 100

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
            
            # 週順にソート
            ng_rates = ng_rates.sort_index()
            
            # 週番号を振り直す
            weeks = range(1, len(ng_rates) + 1)
            
            ax.plot(weeks, ng_rates.values, label=f'品番 {product}', marker='o')
            
            # 7日間未満の週にテキストを表示
            for i, (week, group) in enumerate(df_product.groupby('週')):
                if group['日時'].dt.dayofweek.nunique() < 7:
                    days = group['日時'].dt.dayofweek.nunique()
                    ax.text(i+1, ng_rates.iloc[i], f'({days})', ha='center', va='bottom')
        
        ax.set_xlabel('週')
        ax.set_ylabel('NG率 [%]')
        ax.set_title(f'{machine}の週ごとのNG率')
        ax.set_xticks(weeks)
        ax.set_xticklabels([f'{w}週目' for w in weeks])
        ax.set_ylim(0, 100)
        ax.legend()
        plt.grid(True)
        
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

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
    total = len(group)
    ng_count = (group['目的変数'] == 1).sum()
    return ng_count / total * 100 if total > 0 else 0

# 稼働時間の取得関数
def get_operation_hours(group):
    if len(group) == 0:
        return "稼働なし"
    start = group['日時'].min().strftime('%H:%M')
    end = group['日時'].max().strftime('%H:%M')
    return f"{start}~{end}"

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
            
            # 稼働時間の取得
            operation_hours = df_product.groupby('週').apply(get_operation_hours).iloc[0]
            
            # プロット
            line, = ax.plot(range(len(ng_rates)), ng_rates.values, label=f'品番 {product} ({operation_hours})', marker='o')
            
            # NG率が7.5%以上の点にテキストを追加
            for i, rate in enumerate(ng_rates):
                if rate >= 7.5:
                    ax.text(i, rate, f'{rate:.1f}%', ha='center', va='bottom')
        
        ax.set_xlabel('週')
        ax.set_ylabel('NG率 [%]')
        ax.set_title(f'{machine}の週別NG率')
        ax.set_xticks(range(len(ng_rates)))
        ax.set_xticklabels([f'{i+1}週目' for i in range(len(ng_rates))])
        ax.set_ylim(0, 100)
        ax.legend()
        plt.grid(True)
        
        # 7日間ない週にテキストを追加
        for i, week in enumerate(ng_rates.index):
            days_in_week = df_machine[df_machine['週'] == week]['日時'].dt.date.nunique()
            if days_in_week < 7:
                ax.text(i, -5, f'({days_in_week})', ha='center')
        
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

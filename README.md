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
    return ng_count / total * 100 if total > 0 else 0, ng_count, total

# 稼働時間の取得関数
def get_operation_hours(group):
    start_time = group['日時'].min().strftime('%H:%M')
    end_time = group['日時'].max().strftime('%H:%M')
    return f"{start_time}~{end_time}"

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
            weeks = [week.strftime('%Y-%W') for week in ng_rates.index]
            
            rates = [rate[0] for rate in ng_rates]
            ax.plot(weeks, rates, label=f'品番 {product}', marker='o')
            
            # NG率が7.5%以上の場合、テキストを表示
            for i, (rate, ng_count, total) in enumerate(ng_rates):
                if rate >= 7.5:
                    ax.annotate(f"{rate:.1f}%\n({ng_count}/{total})", (weeks[i], rate),
                                xytext=(0, 10), textcoords='offset points', ha='center')
            
            # 稼働時間を取得
            operation_hours = df_product.groupby('週').apply(get_operation_hours)
            
            # 凡例に稼働時間を追加
            ax.plot([], [], ' ', label=f'品番 {product} (稼働時間: {", ".join(operation_hours)})')
        
        ax.set_xlabel('週')
        ax.set_ylabel('NG率 [%]')
        ax.set_title(f'{machine}の週ごとNG率')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # 7日間ない週にテキストを追加
        for i, week in enumerate(weeks):
            if ng_rates.iloc[i][2] < 7:
                ax.text(i, -5, f'({ng_rates.iloc[i][2]})', ha='center')
        
        plt.tight_layout()
        
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

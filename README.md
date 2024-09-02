# データの前処理
df['日時'] = pd.to_datetime(df['日時'])
df['週'] = (df['日時'] - df['日時'].min()).dt.days // 7 + 1

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

# 稼働時間の計算関数
def get_operation_hours(group):
    start_time = group['日時'].min().strftime('%H:%M')
    end_time = group['日時'].max().strftime('%H:%M')
    return f"{start_time}~{end_time}"

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_週ごとの偏り_全鋳造機_{current_time}.pdf')

with PdfPages(pdf_filename) as pdf:
    # 鋳造機名ごとにプロットを作成
    for machine in df['鋳造機名'].unique():
        plt.figure(figsize=(15, 10))
        
        # 鋳造機名でフィルタリング
        df_machine = df[df['鋳造機名'] == machine]
        
        # 品番ごとにNG率を計算し、プロット
        for product in df_machine['品番'].unique():
            df_product = df_machine[df_machine['品番'] == product]
            ng_rates = df_product.groupby('週').apply(calculate_ng_rate)
            
            # 稼働時間の計算
            operation_hours = df_product.groupby('週').apply(get_operation_hours)
            
            plt.plot(ng_rates.index, ng_rates.values, label=f'品番 {product}(稼働時間: {operation_hours.iloc[0]})', marker='o')
            
            # NG率が7.5%以上の場合、テキストで表示
            for week, rate in ng_rates.items():
                if rate >= 7.5:
                    plt.text(week, rate, f'{rate:.2f}%', ha='center', va='bottom')
        
        plt.xlabel('週')
        plt.ylabel('NG率 [%]')
        plt.title(f'{machine}の週別NG率')
        plt.xticks(range(1, df_machine['週'].max() + 1))
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        
        # 7日間ない週の表示
        for week in range(1, df_machine['週'].max() + 1):
            week_data = df_machine[df_machine['週'] == week]
            days_in_week = (week_data['日時'].max() - week_data['日時'].min()).days + 1
            if days_in_week < 7:
                plt.text(week, plt.ylim()[1], f'({days_in_week})', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # PDFに追加
        pdf.savefig()
        
        # PNGとして保存
        png_filename = os.path.join(output_dir, f'vis_週ごとの偏り_{machine}_{current_time}.png')
        plt.savefig(png_filename)
        
        # グラフを表示（フラグがTrueの場合）
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    print(f"全鋳造機のグラフをPDFに保存しました: {pdf_filename}")

print("各鋳造機のグラフをPNGに保存しました。")

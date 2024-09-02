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

# NG率とNG数の計算関数
def calculate_ng_rate_and_count(group):
    total = len(group)
    ng_count = group['目的変数'].sum()
    return pd.Series({'NG率': (ng_count / total) * 100 if total > 0 else 0,
                      'NG数': ng_count,
                      '全データ数': total})

# 週ごとのデータ数を計算する関数
def count_days_in_week(group):
    return group['日時'].dt.date.nunique()

# 稼働時間を計算する関数
def calculate_operation_hours(group):
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
            ng_data = df_product.groupby('週').apply(calculate_ng_rate_and_count).reset_index()
            
            ax.plot(ng_data['週'], ng_data['NG率'], label=f'品番 {product}', marker='o')
            
            # NG率が7.5%以上の場合、NG数を表示
            for _, row in ng_data[ng_data['NG率'] >= 7.5].iterrows():
                ax.annotate(f"{int(row['NG数'])}/{int(row['全データ数'])}", 
                            (row['週'], row['NG率']), 
                            xytext=(0, 10), textcoords='offset points', 
                            ha='center', va='bottom')
            
            # 稼働時間を計算
            operation_hours = df_product.groupby('週').apply(calculate_operation_hours).reset_index()
            operation_hours_str = ', '.join([f"{row['週']}週目:{row[0]}" for _, row in operation_hours.iterrows()])
            
            # 凡例のラベルに稼働時間を追加
            ax.lines[-1].set_label(f'品番 {product} (稼働時間: {operation_hours_str})')
        
        # 週ごとのデータ数を計算
        days_in_week = df_machine.groupby('週').apply(count_days_in_week)
        
        ax.set_xlabel('週')
        ax.set_ylabel('NG率 [%]')
        ax.set_title(f'{machine}の週別NG率')
        ax.set_xticks(range(1, df_machine['週'].max() + 1))
        ax.set_ylim(0, 100)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True)
        
        # 7日未満の週にテキストを追加
        for week, days in days_in_week.items():
            if days < 7:
                ax.text(week, ax.get_ylim()[1], f'({days})', ha='center', va='bottom')
        
        # PDFに追加
        pdf.savefig(fig, bbox_inches='tight')
        
        # PNGとして保存
        png_filename = os.path.join(output_dir, f'vis_週ごとの偏り_{machine}_{current_time}.png')
        plt.savefig(png_filename, bbox_inches='tight')
        
        # グラフを表示（フラグがTrueの場合）
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    print(f"全鋳造機のグラフをPDFに保存しました: {pdf_filename}")

print("各鋳造機のグラフをPNGに保存しました。")

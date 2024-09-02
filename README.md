# データの前処理
df['日時'] = pd.to_datetime(df['日時'])
df['日付'] = df['日時'].dt.date

# 出力ディレクトリの作成
output_dir = r'..\data\output\eda\NG数の時系列の偏り\日ごとの偏り'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# グラフを表示するかどうかのフラグ
show_plots = False  # Trueにするとグラフを表示、Falseにすると表示しない

# NG率の計算関数
def calculate_ng_rate(group):
    total = group.shape[0]
    if total == 0:
        return pd.Series({'ng_count': 0, 'total': 0, 'ng_rate': None})
    ng_count = group[group['目的変数'] == 1].shape[0]
    ng_rate = (ng_count / total) * 100
    return pd.Series({'ng_count': ng_count, 'total': total, 'ng_rate': ng_rate})

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_日ごとの偏り_全鋳造機_{current_time}.pdf')

with PdfPages(pdf_filename) as pdf:
    # 鋳造機名ごとにプロットを作成
    for machine in df['鋳造機名'].unique():
        df_machine = df[df['鋳造機名'] == machine]
        
        # 日付の範囲を取得
        start_date = df_machine['日付'].min()
        end_date = df_machine['日付'].max()
        
        # 1週間ごとにプロットを作成
        current_date = start_date
        while current_date <= end_date:
            week_end = current_date + timedelta(days=6)
            
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # 品番ごとにNG率を計算し、プロット
            for product in df_machine['品番'].unique():
                df_product = df_machine[df_machine['品番'] == product]
                
                # 週の全日付を生成
                date_range = pd.date_range(current_date, week_end)
                
                # NG率を計算し、全日付に対応するデータフレームを作成
                ng_rates = df_product[(df_product['日付'] >= current_date) & (df_product['日付'] <= week_end)].groupby('日付').apply(calculate_ng_rate)
                ng_rates_full = pd.DataFrame(index=date_range, columns=['ng_count', 'total', 'ng_rate'])
                ng_rates_full.update(ng_rates)
                
                # プロット用のデータを準備
                x_values = ng_rates_full.index
                y_values = ng_rates_full['ng_rate'].tolist()
                
                # Noneの値を含めてプロット
                line, = ax.plot(x_values, y_values, label=f'品番 {product}', marker='o')
                
                # NG率が1.0%以上の点にテキストを追加
                for x, y, ng, total in zip(x_values, y_values, ng_rates_full['ng_count'], ng_rates_full['total']):
                    if pd.notnull(y) and y >= 1.0:
                        ax.annotate(f"{ng}/{total}", (x, y), xytext=(0, 10), 
                                    textcoords='offset points', ha='center', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                                    fontsize=12)  # フォントサイズを12に設定
            
            ax.set_xlabel('日付', fontsize=14)
            ax.set_ylabel('NG率 [%]', fontsize=14)
            ax.set_title(f'{machine}の日ごとNG率 ({current_date} - {week_end})', fontsize=16)
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_locator(plt.MultipleLocator(10))  # Y軸の目盛りを10%ごとに設定
            ax.set_yticklabels([f'{int(x)}%' for x in ax.get_yticks()], fontsize=12)  # Y軸のラベルを設定
            ax.legend(fontsize=12)
            plt.grid(True)
            
            # x軸の日付フォーマットを設定
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # PDFに追加
            pdf.savefig(fig)
            
            # PNGとして保存
            png_filename = os.path.join(output_dir, f'vis_日ごとの偏り_{machine}_{current_date.strftime("%Y%m%d")}_{current_time}.png')
            plt.savefig(png_filename)
            
            # グラフを表示（フラグがTrueの場合）
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
            
            current_date = week_end + timedelta(days=1)
    
    print(f"全鋳造機のグラフをPDFに保存しました: {pdf_filename}")

print("各鋳造機のグラフをPNGに保存しました。")

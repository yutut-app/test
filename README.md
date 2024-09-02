# PDFファイルの作成
daily_pdf_filename = os.path.join(DAILY_OUTPUT_DIR, f'vis_日ごとの偏り_全鋳造機_{CURRENT_TIME}.pdf')

with PdfPages(daily_pdf_filename) as pdf:
    for machine in df['鋳造機名'].unique():
        df_machine = df[df['鋳造機名'] == machine]
        
        start_date = df_machine['日付'].min()
        end_date = df_machine['日付'].max()
        
        # 最初の月曜日を見つける
        while start_date.weekday() != 0:  # 0 は月曜日
            start_date -= timedelta(days=1)
        
        current_date = start_date
        while current_date <= end_date:
            week_end = current_date + timedelta(days=6)
            
            # この週のデータがあるか確認
            week_data = df_machine[(df_machine['日付'] >= current_date) & (df_machine['日付'] <= week_end)]
            if week_data.empty:
                current_date = week_end + timedelta(days=1)
                continue
            
            fig, ax = plt.subplots(figsize=(15, 10))
            
            for product in df_machine['品番'].unique():
                df_product = df_machine[df_machine['品番'] == product]
                ng_rates = df_product[(df_product['日付'] >= current_date) & (df_product['日付'] <= week_end)].groupby('日付').apply(calculate_ng_rate)
                
                valid_data = ng_rates.dropna()
                x_values = valid_data.index
                y_values = [i[2] for i in valid_data.values]
                
                ax.plot(x_values, y_values, label=f'品番 {product}', marker='o', color=PRODUCT_COLOR_MAP.get(product, 'gray'))
                
                for x, y, (ng, total, _) in zip(x_values, y_values, valid_data.values):
                    if y >= 1.0:
                        ax.annotate(f"{ng}/{total}", (x, y), xytext=(0, 10), 
                                    textcoords='offset points', ha='center', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                                    fontsize=12)
            
            ax.set_xlabel('日付', fontsize=14)
            ax.set_ylabel('NG率 [%]', fontsize=14)
            ax.set_title(f'{machine}の日ごとNG率 ({current_date.strftime("%Y/%m/%d")} - {week_end.strftime("%Y/%m/%d")})', fontsize=16)
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_locator(plt.MultipleLocator(10))
            ax.set_yticklabels([f'{int(x)}%' for x in ax.get_yticks()], fontsize=12)
            ax.legend(fontsize=12)
            plt.grid(True)
            
            ax.set_xlim(current_date, week_end)
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n(%a)'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
            
            pdf.savefig(fig)
            
            daily_png_filename = os.path.join(DAILY_OUTPUT_DIR, f'vis_日ごとの偏り_{machine}_{current_date.strftime("%Y%m%d")}_{CURRENT_TIME}.png')
            plt.savefig(daily_png_filename)
            
            if SHOW_PLOTS_DAILY:
                plt.show()
            else:
                plt.close(fig)
            
            current_date = week_end + timedelta(days=1)
    
    print(f"全鋳造機の日ごとの偏りグラフをPDFに保存しました: {daily_pdf_filename}")

print("各鋳造機の日ごとの偏りグラフをPNGに保存しました。")

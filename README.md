# 出力ディレクトリの設定
HOURLY_OUTPUT_DIR = f"{OUTPUT_DIR}/時間の偏り"
os.makedirs(HOURLY_OUTPUT_DIR, exist_ok=True)

# PDFファイルの作成
hourly_pdf_filename = os.path.join(HOURLY_OUTPUT_DIR, f'vis_時間の偏り_全鋳造機_{CURRENT_TIME}.pdf')

with PdfPages(hourly_pdf_filename) as pdf:
    for machine in df['鋳造機名'].unique():
        df_machine = df[df['鋳造機名'] == machine]
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        for product in df_machine['品番'].unique():
            df_product = df_machine[df_machine['品番'] == product]
            ng_rates = df_product.groupby('時間').apply(calculate_ng_rate, include_groups=False)
            
            valid_data = ng_rates.dropna()
            x_values = valid_data.index
            y_values = [i[2] for i in valid_data.values]
            
            ax.plot(x_values, y_values, label=f'品番 {product}', marker='o', color=PRODUCT_COLOR_MAP.get(product, 'gray'))
            
            for x, y, (ng, total, _) in zip(x_values, y_values, valid_data.values):
                if y >= 7.5:
                    ax.annotate(f"{ng}/{total}", (x, y), xytext=(0, 10), 
                                textcoords='offset points', ha='center', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                                fontsize=12)
        
        ax.set_xlabel('時間 (時)', fontsize=14)
        ax.set_ylabel('NG率 [%]', fontsize=14)
        ax.set_title(f'{machine}の時間帯別NG率', fontsize=16)
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24))
        ax.set_ylim(0, 100)
        
        # Y軸のティックを固定
        y_ticks = range(0, 101, 10)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y}%' for y in y_ticks], fontsize=12)
        
        ax.legend(fontsize=12)
        plt.grid(True)
        
        pdf.savefig(fig)
        
        hourly_png_filename = os.path.join(HOURLY_OUTPUT_DIR, f'vis_時間の偏り_{machine}_{CURRENT_TIME}.png')
        plt.savefig(hourly_png_filename)
        
        if SHOW_PLOTS_HOURLY:
            plt.show()
        else:
            plt.close(fig)

    print(f"全鋳造機の時間の偏りグラフをPDFに保存しました: {hourly_pdf_filename}")

print("各鋳造機の時間の偏りグラフをPNGに保存しました。")

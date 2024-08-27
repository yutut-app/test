# グラフを表示するかどうかのフラグ
show_plots = True  # Trueにするとグラフを表示、Falseにすると表示しない

# 全鋳造機名のデータを使用してNG率を計算
ng_rate_by_product_machine = df.groupby(['品番', '鋳造機名'])['目的変数'].agg(['count', calculate_ng_rate]).reset_index()
ng_rate_by_product_machine.columns = ['品番', '鋳造機名', 'データ数', 'NG率']

# 総データ数を計算
total_count = len(df)

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_全鋳造機名の品番ごとNG率_{current_time}.pdf')

with PdfPages(pdf_filename) as pdf:
    fig, ax = plt.subplots(figsize=(15, 8))
    bars = sns.barplot(x='品番', y='NG率', hue='鋳造機名', data=ng_rate_by_product_machine, ax=ax)
    plt.title(f'全鋳造機名の品番ごとの渦流探傷NG率 (n={total_count})')
    plt.xlabel('品番')
    plt.ylabel('NG率 [%]')
    plt.ylim(0, 100)  # Y軸の最大値を100%に設定
    
    # 各棒グラフの上に値とテキストを表示
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        product = ng_rate_by_product_machine['品番'].iloc[i // len(df['鋳造機名'].unique())]
        machine = ng_rate_by_product_machine['鋳造機名'].iloc[i % len(df['鋳造機名'].unique())]
        data = ng_rate_by_product_machine[(ng_rate_by_product_machine['品番'] == product) & 
                                          (ng_rate_by_product_machine['鋳造機名'] == machine)]
        if not data.empty:
            total_count = data['データ数'].values[0]
            ng_count = int(total_count * height / 100)
            text = f"{height:.2f}%\n({ng_count}/{total_count})"
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    text,
                    ha='center', va='bottom', fontsize=8)
    
    plt.legend(title='鋳造機名', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

print(f"全鋳造機名のグラフをPDFに保存しました: {pdf_filename}")

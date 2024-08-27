# NG率の計算
def calculate_ng_rate(group):
    total = len(group)
    ng_count = (group == 1).sum()
    return ng_count, total
    
# 品番ごとのNG率を計算
ng_rate_by_product = df.groupby(['鋳造機名', '品番'])['目的変数'].agg(calculate_ng_rate).reset_index()
ng_rate_by_product.columns = ['鋳造機名', '品番', 'NG_count', 'Total_count']
ng_rate_by_product['NG率'] = ng_rate_by_product['NG_count'] / ng_rate_by_product['Total_count'] * 100

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_鋳造機名ごとの品番ごとNG率_{current_time}.pdf')
png_filename = os.path.join(output_dir, f'vis_鋳造機名ごとの品番ごとNG率_{current_time}.png')

fig, ax = plt.subplots(figsize=(15, 8))
bars = sns.barplot(x='品番', y='NG率', hue='鋳造機名', data=ng_rate_by_product, ax=ax)

plt.title('鋳造機名ごとの品番ごとの渦流探傷NG率')
plt.xlabel('品番')
plt.ylabel('NG率 [%]')
plt.ylim(0, 100)  # Y軸の最大値を100%に設定

# 各棒グラフの上に値とテキストを表示
for i, bar in enumerate(bars.patches):
    height = bar.get_height()
    machine = ng_rate_by_product['鋳造機名'].iloc[i // 4]
    product = ng_rate_by_product['品番'].iloc[i % 4]
    ng_count = ng_rate_by_product.loc[(ng_rate_by_product['鋳造機名'] == machine) & 
                                      (ng_rate_by_product['品番'] == product), 'NG_count'].values[0]
    total_count = ng_rate_by_product.loc[(ng_rate_by_product['鋳造機名'] == machine) & 
                                         (ng_rate_by_product['品番'] == product), 'Total_count'].values[0]
    
    if machine == '2号機' and product == '4':
        text = "0.00% (0/0)"
    else:
        text = f"{height:.2f}% ({ng_count}/{total_count})"
    
    ax.text(bar.get_x() + bar.get_width()/2., height,
            text,
            ha='center', va='bottom')

plt.tight_layout()

# PDFに保存
with PdfPages(pdf_filename) as pdf:
    pdf.savefig(fig)

# PNGに保存
plt.savefig(png_filename)

if show_plots:
    plt.show()
else:
    plt.close(fig)

print(f"グラフをPDFに保存しました: {pdf_filename}")
print(f"グラフをPNGに保存しました: {png_filename}")

# NG率の計算
def calculate_ng_rate(group):
    total = len(group)
    ng_count = (group == 1).sum()
    return ng_count, total, (ng_count / total) * 100 if total > 0 else 0

# 品番ごとのNG率を計算
ng_rate_by_product = df.groupby(['鋳造機名', '品番'])['目的変数'].apply(calculate_ng_rate).reset_index()
ng_rate_by_product.columns = ['鋳造機名', '品番', 'NG_Count_Total_Rate']
ng_rate_by_product[['NG_Count', 'Total_Count', 'NG率']] = pd.DataFrame(ng_rate_by_product['NG_Count_Total_Rate'].tolist(), index=ng_rate_by_product.index)

# グラフの作成
fig, ax = plt.subplots(figsize=(15, 8))

# 棒グラフの幅
bar_width = 0.35

# 鋳造機名と品番の組み合わせを取得
machines = df['鋳造機名'].unique()
products = df['品番'].unique()

# 棒グラフの位置を計算
x = np.arange(len(products))

# 各鋳造機名ごとに棒グラフを作成
for i, machine in enumerate(machines):
    data = ng_rate_by_product[ng_rate_by_product['鋳造機名'] == machine]
    bars = ax.bar(x + i*bar_width, data['NG率'], bar_width, label=machine)

    # 各棒グラフの上に値とテキストを表示
    for bar in bars:
        height = bar.get_height()
        product = products[int(bar.get_x() / bar_width)]
        row = data[data['品番'] == product]
        if not row.empty and row['Total_Count'].values[0] > 0:
            ng_count = row['NG_Count'].values[0]
            total_count = row['Total_Count'].values[0]
            text = f"{height:.2f}%\n({ng_count}/{total_count})"
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    text,
                    ha='center', va='bottom')

plt.title(f'鋳造機名ごとの品番ごとの渦流探傷NG率')
plt.xlabel('品番')
plt.ylabel('NG率 [%]')
plt.ylim(0, 100)  # Y軸の最大値を100%に設定
plt.xticks(x + bar_width / 2, products)
plt.legend()

plt.tight_layout()

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_鋳造機名ごとの品番ごとNG率_{current_time}.pdf')
plt.savefig(pdf_filename)

# PNGファイルを作成
png_filename = os.path.join(output_dir, f'vis_鋳造機名ごとの品番ごとNG率_{current_time}.png')
plt.savefig(png_filename)

if show_plots:
    plt.show()
else:
    plt.close(fig)

print(f"グラフをPDFに保存しました: {pdf_filename}")
print(f"グラフをPNGに保存しました: {png_filename}")

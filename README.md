# NG率の計算
def calculate_ng_rate(group):
    total = len(group)
    ng_count = (group == 1).sum()
    return (ng_count, total, ng_count / total * 100 if total > 0 else 0)

# 鋳造機名と品番の組み合わせごとにNG率を計算
ng_rate_by_machine_product = df.groupby(['鋳造機名', '品番'])['目的変数'].agg(calculate_ng_rate).reset_index()
ng_rate_by_machine_product.columns = ['鋳造機名', '品番', 'NG_データ']
ng_rate_by_machine_product[['NG回数', '全回数', 'NG率']] = pd.DataFrame(ng_rate_by_machine_product['NG_データ'].tolist(), index=ng_rate_by_machine_product.index)
ng_rate_by_machine_product = ng_rate_by_machine_product.drop('NG_データ', axis=1)

# グラフの作成
fig, ax = plt.subplots(figsize=(15, 8))

# 鋳造機名ごとに色を変えて棒グラフを作成
colors = ['blue', 'red']
machines = ng_rate_by_machine_product['鋳造機名'].unique()
width = 0.35  # バーの幅

for i, machine in enumerate(machines):
    data = ng_rate_by_machine_product[ng_rate_by_machine_product['鋳造機名'] == machine]
    x = np.arange(len(data['品番']))
    rects = ax.bar(x + i*width, data['NG率'], width, label=machine, color=colors[i])
    
    # 各棒グラフの上に値とテキストを表示
    for rect, ng_count, total_count in zip(rects, data['NG回数'], data['全回数']):
        height = rect.get_height()
        if total_count > 0:  # 2号機の品番4のケースを除外
            text = f"{height:.2f}%\n({ng_count}/{total_count})"
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    text,
                    ha='center', va='bottom')

ax.set_ylabel('NG率 [%]')
ax.set_xlabel('品番')
ax.set_title('鋳造機名ごとの品番別NG率')
ax.set_xticks(np.arange(len(data['品番'])) + width / 2)
ax.set_xticklabels(data['品番'])
ax.legend()
ax.set_ylim(0, 100)  # Y軸の最大値を100%に設定

plt.tight_layout()

# PDFとPNGで保存
pdf_filename = os.path.join(output_dir, f'vis_鋳造機名ごとの品番別NG率_{current_time}.pdf')
png_filename = os.path.join(output_dir, f'vis_鋳造機名ごとの品番別NG率_{current_time}.png')

plt.savefig(pdf_filename)
plt.savefig(png_filename)

if show_plots:
    plt.show()
else:
    plt.close(fig)

print(f"グラフをPDFに保存しました: {pdf_filename}")
print(f"グラフをPNGに保存しました: {png_filename}")

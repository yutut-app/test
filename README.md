# 出力ディレクトリの作成
output_dir = r'..\data\output\eda\鋳造条件と品質の分析\鋳造機ごとの比較'
pdf_output_dir = os.path.join(output_dir, 'pdf')
png_output_dir = os.path.join(output_dir, 'png')
os.makedirs(pdf_output_dir, exist_ok=True)
os.makedirs(png_output_dir, exist_ok=True)

# 品番ごとにPNGファイルを作成
for product in unique_products:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, machine in enumerate(machines):
        data = ng_rate_by_machine_product[(ng_rate_by_machine_product['鋳造機名'] == machine) & 
                                          (ng_rate_by_machine_product['品番'] == product)]
        if not data.empty:
            rects = ax.bar(machine, data['NG率'].values[0], width, color=colors[i])
            
            # 各棒グラフの上に値とテキストを表示
            height = data['NG率'].values[0]
            ng_count = data['NG回数'].values[0]
            total_count = data['全回数'].values[0]
            if total_count > 0:
                text = f"{height:.2f}%\n({ng_count}/{total_count})"
                ax.text(i, height, text, ha='center', va='bottom')

    ax.set_ylabel('NG率 [%]')
    ax.set_title(f'品番 {product} の鋳造機名ごとのNG率')
    ax.set_ylim(0, 100)  # Y軸の最大値を100%に設定

    plt.tight_layout()

    # PNGに保存
    png_filename = os.path.join(png_output_dir, f'vis_品番{product}の鋳造機名ごとのNG率_{current_time}.png')
    plt.savefig(png_filename)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    print(f"品番 {product} のグラフをPNGに保存しました: {png_filename}")

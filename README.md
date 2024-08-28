

# グラフを表示するかどうかのフラグ
show_plots = True  # Trueにするとグラフを表示、Falseにすると表示しない

# 品番を1号機に固定
product = '1号機'

# PDFファイルを作成
pdf_filename = os.path.join(pdf_output_dir, f'vis_{product}と鋳造機名の真空圧３（絶対圧）の関係_{current_time}.pdf')

with PdfPages(pdf_filename) as pdf:
    # 真空圧３（絶対圧）のみをプロット
    condition = '真空圧３（絶対圧）'
    
    # プロットの作成
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 現在の品番のデータのみをフィルタリング
    df_product = df[df['品番'] == product]
    
    # OKとNGのデータを分離
    df_ok = df_product[df_product['目的変数'] == 0]
    df_ng = df_product[df_product['目的変数'] == 1]
    
    # OKのデータをプロット（透明度を下げる）
    sns.stripplot(data=df_ok, x=condition, y='鋳造機名', color='blue', alpha=0.3, 
                  jitter=True, size=5, ax=ax, order=sorted(df_product['鋳造機名'].unique()))
    
    # NGのデータをプロット（前面に、大きく、透明度を上げる）
    sns.stripplot(data=df_ng, x=condition, y='鋳造機名', color='orange', alpha=1.0, 
                  jitter=True, size=10, ax=ax, order=sorted(df_product['鋳造機名'].unique()))
    
    # タイトルと軸ラベルの設定
    plt.title(f'{product}の鋳造機名と{condition}の関係')
    plt.xlabel(condition)
    plt.ylabel('鋳造機名')
    
    # 凡例の設定
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='OK (0)',
                              markerfacecolor='blue', markersize=10, alpha=0.3),
                       Line2D([0], [0], marker='o', color='w', label='NG (1)',
                              markerfacecolor='orange', markersize=15)]
    ax.legend(handles=legend_elements, title='目的変数')
    
    # グラフの調整
    plt.tight_layout()
    
    # PDFに追加
    pdf.savefig(fig)
    
    # PNGとして保存
    png_filename = os.path.join(png_output_dir, f'vis_{product}の鋳造機名と{condition}の関係_{current_time}.png')
    plt.savefig(png_filename)
    
    # グラフを表示（フラグがTrueの場合）
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

print(f"{product}のグラフをPDFに保存しました: {pdf_filename}")
print(f"{product}の真空圧３（絶対圧）のグラフをPNGに保存しました: {png_output_dir}")

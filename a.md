# jitterを追加する関数
def add_jitter(values, jitter_amount=0.3):
    return values + np.random.uniform(-jitter_amount, jitter_amount, len(values))

# グラフを表示するかどうかのフラグ
show_plots = True  # Trueにするとグラフを表示、Falseにすると表示しない

# PDFファイルを開く
with PdfPages(pdf_filename) as pdf:
    # 各鋳造条件に対してプロットを作成
    for condition in casting_condition_columns:
        # プロットの作成
        fig, ax = plt.subplots(figsize=(12, 3))
        
        # OKとNGのデータを分離
        df_ok = df[df['目的変数'] == 0]
        df_ng = df[df['目的変数'] == 1]
        
        # OKのデータをプロット（透明度を下げる）
        sns.stripplot(data=df_ok, x=condition, y='鋳造機名', color='blue', alpha=0.3, 
                      jitter=True, size=5, ax=ax)
        
        # NGのデータをプロット（前面に、大きく、透明度を上げる）
        sns.stripplot(data=df_ng, x=condition, y='鋳造機名', color='red', alpha=1.0, 
                      jitter=True, size=10, ax=ax)
        
        # タイトルと軸ラベルの設定
        plt.title(f'鋳造機名と{condition}の関係')
        plt.xlabel(condition)
        plt.ylabel('鋳造機名')
        
        # 凡例の設定
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='OK (0)',
                                  markerfacecolor='blue', markersize=10, alpha=0.3),
                           Line2D([0], [0], marker='o', color='w', label='NG (1)',
                                  markerfacecolor='red', markersize=15)]
        ax.legend(handles=legend_elements, title='目的変数')
        
        # グラフの調整
        plt.tight_layout()
        
        # PDFに追加
        pdf.savefig(fig)
        
        # グラフを表示（フラグがTrueの場合）
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

print(f"グラフをPDFに保存しました: {pdf_filename}")

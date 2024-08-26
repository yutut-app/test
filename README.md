# 出力ディレクトリの作成
output_dir = r'..\data\output\eda'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# グラフを表示するかどうかのフラグ
show_plots = True  # Trueにするとグラフを表示、Falseにすると表示しない

# 鋳造機名ごとにプロットを作成
for machine in df['鋳造機名'].unique():
    # PDFファイルを作成
    pdf_filename = os.path.join(output_dir, f'vis_{machine}と品番の鋳造条件の関係_{current_time}.pdf')
    
    with PdfPages(pdf_filename) as pdf:
        # 各鋳造条件に対してプロットを作成
        for condition in casting_condition_columns:
            # プロットの作成
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 現在の鋳造機名のデータのみをフィルタリング
            df_machine = df[df['鋳造機名'] == machine]
            
            # OKとNGのデータを分離
            df_ok = df_machine[df_machine['目的変数'] == 0]
            df_ng = df_machine[df_machine['目的変数'] == 1]
            
            # OKのデータをプロット（透明度を下げる）
            sns.stripplot(data=df_ok, x=condition, y='品番', color='blue', alpha=0.3, 
                          jitter=True, size=5, ax=ax, order=sorted(df_machine['品番'].unique()))
            
            # NGのデータをプロット（前面に、大きく、透明度を上げる）
            sns.stripplot(data=df_ng, x=condition, y='品番', color='orange', alpha=1.0, 
                          jitter=True, size=10, ax=ax, order=sorted(df_machine['品番'].unique()))
            
            # タイトルと軸ラベルの設定
            plt.title(f'{machine}の品番と{condition}の関係')
            plt.xlabel(condition)
            plt.ylabel('品番')
            
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
            
            # グラフを表示（フラグがTrueの場合）
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

    print(f"{machine}のグラフをPDFに保存しました: {pdf_filename}")

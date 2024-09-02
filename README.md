# データの前処理
df['日時'] = pd.to_datetime(df['日時'])
df['週'] = df['日時'].dt.to_period('W').astype(str)

# 出力ディレクトリの作成
output_dir = r'..\data\output\eda\NG数の時系列の偏り\週ごとの偏り'
os.makedirs(output_dir, exist_ok=True)

# 現在の日時を取得（ファイル名用）
current_time = datetime.now().strftime("%y%m%d%H%M")

# グラフを表示するかどうかのフラグ
show_plots = False  # Trueにするとグラフを表示、Falseにすると表示しない

# NG率の計算関数
def calculate_ng_rate(group):
    total = group.shape[0]
    if total == 0:
        return None  # データが無い場合はNoneを返す
    ng_count = group[group['目的変数'] == 1].shape[0]
    return (ng_count / total) * 100 if total > 0 else None

# PDFファイルを作成
pdf_filename = os.path.join(output_dir, f'vis_週ごとの偏り_全鋳造機_{current_time}.pdf')

with PdfPages(pdf_filename) as pdf:
    # 鋳造機名ごとにプロットを作成
    for machine in df['鋳造機名'].unique():
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 鋳造機名でフィルタリング
        df_machine = df[df['鋳造機名'] == machine]
        
        # 品番ごとにNG率を計算し、プロット
        for product in df_machine['品番'].unique():
            df_product = df_machine[df_machine['品番'] == product]
            ng_rates = df_product.groupby('週').apply(calculate_ng_rate)
            
            # Noneの値（データ無し）を除外してプロット
            valid_data = ng_rates.dropna()
            ax.plot(valid_data.index, valid_data.values, label=f'品番 {product}', marker='o')
        
        ax.set_xlabel('週')
        ax.set_ylabel('NG率 [%]')
        ax.set_title(f'{machine}の週ごとNG率')
        ax.set_ylim(0, 100)
        ax.legend()
        plt.grid(True)
        
        # x軸ラベルの回転
        plt.xticks(rotation=45, ha='right')
        
        # レイアウトの調整
        plt.tight_layout()
        
        # PDFに追加
        pdf.savefig(fig)
        
        # PNGとして保存
        png_filename = os.path.join(output_dir, f'vis_週ごとの偏り_{machine}_{current_time}.png')
        plt.savefig(png_filename)
        
        # グラフを表示（フラグがTrueの場合）
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    print(f"全鋳造機のグラフをPDFに保存しました: {pdf_filename}")

print("各鋳造機のグラフをPNGに保存しました。")

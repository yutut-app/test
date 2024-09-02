def calculate_ng_rate(group):
    """グループ内のNG率を計算する。

    Args:
        group (pandas.DataFrame): 計算対象のデータグループ

    Returns:
        tuple: (NG数, 総数, NG率) のタプル。データがない場合はNone。
    """
    total = len(group)
    if total == 0:
        return None
    ng_count = (group['目的変数'] == 1).sum()
    ng_rate = (ng_count / total) * 100 if total > 0 else None
    return ng_count, total, ng_rate

def plot_hourly_ng_rate(df, machine, output_dir, current_time, color_map, show_plots=False):
    """時間帯別NG率をプロットし、PDFとPNGで保存する。

    Args:
        df (pandas.DataFrame): 分析対象のデータフレーム
        machine (str): 鋳造機名
        output_dir (str): 出力ディレクトリのパス
        current_time (str): ファイル名用の現在時刻文字列
        color_map (dict): 品番ごとの色マッピング
        show_plots (bool, optional): プロットを表示するかどうか。デフォルトはFalse。
    """
    df_machine = df[df['鋳造機名'] == machine]
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for product in df_machine['品番'].unique():
        df_product = df_machine[df_machine['品番'] == product]
        ng_rates = df_product.groupby('時間').apply(calculate_ng_rate)
        
        valid_data = ng_rates.dropna()
        x_values = valid_data.index
        y_values = [i[2] for i in valid_data.values]
        
        ax.plot(x_values, y_values, label=f'品番 {product}', marker='o',
                color=color_map.get(product, 'gray'))
        
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
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels(range(0, 24), fontsize=12)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.set_yticklabels([f'{int(x)}%' for x in ax.get_yticks()], fontsize=12)
    ax.legend(fontsize=12)
    plt.grid(True)
    
    pdf_filename = os.path.join(output_dir, f'vis_時間の偏り_{machine}_{current_time}.pdf')
    png_filename = os.path.join(output_dir, f'vis_時間の偏り_{machine}_{current_time}.png')
    
    plt.savefig(pdf_filename)
    plt.savefig(png_filename)
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

def analyze_hourly_ng_rate(df, output_dir, color_map, show_plots=False):
    """全鋳造機の時間帯別NG率を分析し、結果を保存する。

    Args:
        df (pandas.DataFrame): 分析対象のデータフレーム
        output_dir (str): 出力ディレクトリのパス
        color_map (dict): 品番ごとの色マッピング
        show_plots (bool, optional): プロットを表示するかどうか。デフォルトはFalse。
    """
    current_time = datetime.now().strftime("%y%m%d%H%M")
    pdf_filename = os.path.join(output_dir, f'vis_時間の偏り_全鋳造機_{current_time}.pdf')
    
    with PdfPages(pdf_filename) as pdf:
        for machine in df['鋳造機名'].unique():
            plot_hourly_ng_rate(df, machine, output_dir, current_time, color_map, show_plots)
            pdf.savefig()
    
    print(f"全鋳造機のグラフをPDFに保存しました: {pdf_filename}")
    print("各鋳造機のグラフをPNGに保存しました。")

# メイン処理
df = pd.read_csv('casting_data.csv')
df['日時'] = pd.to_datetime(df['日時'])
df['時間'] = df['日時'].dt.hour

output_dir = r'..\data\output\eda\NG数の時系列の偏り\時間の偏り'
os.makedirs(output_dir, exist_ok=True)

color_map = {2: 'red', 4: 'blue', 5: 'green', 6: 'purple'}

analyze_hourly_ng_rate(df, output_dir, color_map)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime
import numpy as np
import pandas as pd

# 出力ディレクトリの設定
defected_data_path = r"../data/output/defect_data"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
pdf_filename = os.path.join(defected_data_path, f'eda_欠陥分析_{current_time}.pdf')

# 独立変数のリストと対応するx軸の設定
independent_vars = ['width', 'height', 'area', 'perimeter', 'eccentricity', 'orientation', 
                    'major_axis_length', 'minor_axis_length', 'solidity', 'extent', 
                    'aspect_ratio', 'max_length']

x_axis_settings = {
    'area': (0, 105, 5),
    'perimeter': (0, 105, 5),
    'eccentricity': (-1.6, 1.6, 0.2),
    'orientation': (0, 1.05, 0.05),
    'major_axis_length': (0, 120, 5),
    'minor_axis_length': (0, 35, 5),
    'solidity': (0, 1.1, 0.05),
    'extent': (0, 1.1, 0.05),
    'aspect_ratio': (0, 80, 5)
}

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Yu Gothic'

# データのサンプリング（処理時間短縮のため）
sample_size = min(10000, len(df))
ng_data = df[df['defect_label'] == 0]  # 欠陥候補（非欠陥）
ok_data = df[df['defect_label'] == 1]  # 欠陥（中巣）

# 欠陥（中巣）データを全て含める
if len(ok_data) < sample_size:
    sampled_ng = ng_data.sample(n=sample_size - len(ok_data), random_state=42)
    df_sampled = pd.concat([ok_data, sampled_ng])
else:
    df_sampled = df.sample(n=sample_size, random_state=42)

# PDFファイルを作成
with PdfPages(pdf_filename) as pdf:
    for var in independent_vars:
        # プロットの作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 欠陥候補（非欠陥）と欠陥（中巣）のデータを分離
        df_ng = df_sampled[df_sampled['defect_label'] == 0]
        df_ok = df_sampled[df_sampled['defect_label'] == 1]
        
        # 欠陥候補（非欠陥）のデータをプロット
        sns.stripplot(data=df_ng, x=var, y='defect_label', color='blue', alpha=0.3, 
                      jitter=True, size=5, ax=ax, dodge=True, zorder=1)
        
        # 欠陥（中巣）のデータをプロット
        sns.stripplot(data=df_ok, x=var, y='defect_label', color='red', alpha=1.0, 
                      jitter=True, size=10, ax=ax, dodge=True, zorder=2)
        
        # タイトルと軸ラベルの設定
        plt.title(f'{var}と欠陥ラベルの関係')
        plt.xlabel(var)
        plt.ylabel('欠陥ラベル')
        
        # x軸の範囲とステップを設定
        if var in x_axis_settings:
            start, end, step = x_axis_settings[var]
            plt.xlim(start, end)
            plt.xticks(np.arange(start, end + step, step))
        
        # y軸の目盛りを設定
        plt.yticks([0, 1], ['欠陥候補（非欠陥） (0)', '欠陥（中巣） (1)'])
        
        # 凡例の設定
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='欠陥候補（非欠陥） (0)', 
                                  markerfacecolor='blue', markersize=10, alpha=0.3),
                           Line2D([0], [0], marker='o', color='w', label='欠陥（中巣） (1)', 
                                  markerfacecolor='red', markersize=15)]
        ax.legend(handles=legend_elements, title='欠陥ラベル')
        
        # グラフの調整
        plt.tight_layout()
        
        # PDFに追加
        pdf.savefig(fig)
        plt.close(fig)

print(f"EDAグラフをPDFに保存しました: {pdf_filename}")

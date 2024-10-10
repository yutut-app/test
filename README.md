import matplotlib.pyplot as plt
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
    'area': (0, 105, 5, 1),
    'perimeter': (0, 105, 5, 1),
    'eccentricity': (-1.6, 1.6, 0.2, 0.05),
    'orientation': (0, 1.05, 0.05, 0.01),
    'major_axis_length': (0, 120, 5, 1),
    'minor_axis_length': (0, 35, 5, 1),
    'solidity': (0, 1.1, 0.05, 0.01),
    'extent': (0, 1.1, 0.05, 0.01),
    'aspect_ratio': (0, 80, 5, 1)
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
        
        # 散布図プロット
        ax.scatter(df_ng[var], np.zeros(len(df_ng)), color='blue', alpha=0.3, s=20, label='欠陥候補（非欠陥） (0)')
        ax.scatter(df_ok[var], np.ones(len(df_ok)), color='red', alpha=1.0, s=40, label='欠陥（中巣） (1)')
        
        # タイトルと軸ラベルの設定
        plt.title(f'{var}と欠陥ラベルの関係')
        plt.xlabel(var)
        plt.ylabel('欠陥ラベル')
        
        # x軸の目盛りを設定
        if var in x_axis_settings:
            start, end, step, minor_step = x_axis_settings[var]
            major_ticks = np.arange(start, end + step, step)
            minor_ticks = np.arange(start, end + minor_step, minor_step)
            ax.set_xlim(start, end)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
        
        # y軸の目盛りを設定
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['欠陥候補（非欠陥） (0)', '欠陥（中巣） (1)'])
        
        # 凡例の設定
        ax.legend(title='欠陥ラベル')
        
        # グラフの調整
        plt.tight_layout()
        
        # PDFに追加
        pdf.savefig(fig)
        plt.close(fig)

print(f"EDAグラフをPDFに保存しました: {pdf_filename}")

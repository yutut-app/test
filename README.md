"""低速（平均）速度とNG率の関係分析モジュール

このモジュールは、鋳造プロセスの低速（平均）速度とNG率の関係を分析し、可視化します。
1号機と2号機を同じグラフに表示し、仮説検証の詳細を調べます。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import os

from time_series_data_prep import (
    df, OUTPUT_DIR, CURRENT_TIME, PRODUCT_COLOR_MAP,
    SHOW_PLOTS_HOURLY, calculate_ng_rate
)

# 出力ディレクトリの設定
HYPOTHESIS_OUTPUT_DIR = f"{OUTPUT_DIR}/仮説検証_低速速度"
os.makedirs(HYPOTHESIS_OUTPUT_DIR, exist_ok=True)

# PDFファイルの作成
hypothesis_pdf_filename = os.path.join(HYPOTHESIS_OUTPUT_DIR, f'vis_低速速度とNG率の関係_{CURRENT_TIME}.pdf')

# binの設定
bins = np.arange(0.294, 0.303, 0.001)

def calculate_ng_rate_for_bin(group):
    total = len(group)
    ng_count = (group['目的変数'] == 1).sum()
    return ng_count / total * 100 if total > 0 else 0

with PdfPages(hypothesis_pdf_filename) as pdf:
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for machine in ['1号機', '2号機']:
        df_machine = df[df['鋳造機名'] == machine]
        
        ng_rates = []
        for left, right in zip(bins[:-1], bins[1:]):
            bin_data = df_machine[(df_machine['低速（平均）速度'] >= left) & (df_machine['低速（平均）速度'] < right)]
            ng_rate = calculate_ng_rate_for_bin(bin_data)
            ng_rates.append(ng_rate)
        
        ax.plot(bins[:-1], ng_rates, label=f'{machine}', marker='o')
    
    ax.set_xlabel('低速（平均）速度', fontsize=14)
    ax.set_ylabel('NG率 [%]', fontsize=14)
    ax.set_title('低速（平均）速度とNG率の関係', fontsize=16)
    ax.set_xlim(0.294, 0.302)
    ax.set_ylim(0, 100)
    
    # Y軸のティックを固定
    y_ticks = range(0, 101, 10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y}%' for y in y_ticks], fontsize=12)
    
    ax.legend(fontsize=12)
    plt.grid(True)
    
    pdf.savefig(fig)
    
    hypothesis_png_filename = os.path.join(HYPOTHESIS_OUTPUT_DIR, f'vis_低速速度とNG率の関係_{CURRENT_TIME}.png')
    plt.savefig(hypothesis_png_filename)
    
    if SHOW_PLOTS_HOURLY:
        plt.show()
    else:
        plt.close(fig)

print(f"低速（平均）速度とNG率の関係グラフをPDFに保存しました: {hypothesis_pdf_filename}")
print(f"低速（平均）速度とNG率の関係グラフをPNGに保存しました: {hypothesis_png_filename}")

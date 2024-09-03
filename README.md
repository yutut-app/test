"""低速（平均）速度とNG率の関係分析モジュール

このモジュールは、鋳造プロセスの低速（平均）速度とNG率の関係を分析し、可視化します。
時系列データの分析準備モジュールで定義された変数や関数を利用します。
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from time_series_data_prep import (
    df, OUTPUT_DIR, CURRENT_TIME, PRODUCT_COLOR_MAP,
    SHOW_PLOTS_HOURLY, calculate_ng_rate
)

# 出力ディレクトリの設定
HYPOTHESIS_OUTPUT_DIR = f"{OUTPUT_DIR}/仮説検証"
os.makedirs(HYPOTHESIS_OUTPUT_DIR, exist_ok=True)

# 低速（平均）速度の列名
LOW_SPEED_COLUMN = '低速(平均)速度'

# binの設定
BINS = np.arange(0.294, 0.303, 0.001)

# NG率の計算
def calculate_ng_rate_for_bin(group):
    total = len(group)
    ng_count = (group['目的変数'] == 1).sum()
    return (ng_count / total) * 100 if total > 0 else 0

# グラフの作成と保存
fig, ax = plt.subplots(figsize=(15, 10))

for machine in ['1号機', '2号機']:
    df_machine = df[df['鋳造機名'] == machine]
    
    # 低速（平均）速度でビニングし、各ビンのNG率を計算
    binned_data = df_machine.groupby(pd.cut(df_machine[LOW_SPEED_COLUMN], bins=BINS))
    ng_rates = binned_data.apply(calculate_ng_rate_for_bin)
    
    # ヒストグラムのプロット
    ax.bar(ng_rates.index.mid, ng_rates.values, 
           width=0.0009, alpha=0.5, label=machine)

ax.set_xlabel('低速（平均）速度', fontsize=14)
ax.set_ylabel('NG率 [%]', fontsize=14)
ax.set_title('低速（平均）速度とNG率の関係', fontsize=16)
ax.set_ylim(0, 100)
ax.legend(fontsize=12)
plt.grid(True)

# Y軸のティックを固定
y_ticks = range(0, 101, 10)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'{y}%' for y in y_ticks], fontsize=12)

# X軸のラベルを調整
plt.xticks(rotation=45, ha='right')

# グラフの保存
hypothesis_filename = os.path.join(HYPOTHESIS_OUTPUT_DIR, f'低速速度_NG率関係_{CURRENT_TIME}')
plt.savefig(f'{hypothesis_filename}.png', bbox_inches='tight')
plt.savefig(f'{hypothesis_filename}.pdf', bbox_inches='tight')

if SHOW_PLOTS_HOURLY:
    plt.show()
else:
    plt.close(fig)

print(f"低速（平均）速度とNG率の関係グラフを保存しました: {hypothesis_filename}")

"""低速（平均）速度とNG率の関係分析モジュール

このモジュールは、鋳造プロセスの低速（平均）速度とNG率の関係を分析し、可視化します。
1号機と2号機の比較を行います。
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from time_series_data_prep import (
    df, OUTPUT_DIR, CURRENT_TIME, calculate_ng_rate
)

# 出力ディレクトリの設定
SPEED_NG_RATE_DIR = f"{OUTPUT_DIR}/低速速度とNG率の関係"
os.makedirs(SPEED_NG_RATE_DIR, exist_ok=True)

# 低速（平均）速度の範囲とビンの設定
speed_bins = np.arange(0.294, 0.303, 0.001)

# 鋳造機ごとのデータを準備
df_machine1 = df[df['鋳造機名'] == '1号機']
df_machine2 = df[df['鋳造機名'] == '2号機']

# NG率の計算
def calculate_ng_rate_for_speed(df, speed_bins):
    df['speed_bin'] = pd.cut(df['低速（平均）速度'], bins=speed_bins)
    grouped = df.groupby('speed_bin').apply(calculate_ng_rate)
    return grouped.apply(lambda x: x[2] if x else 0)

# ヒストグラムのデータ準備
ng_rates_machine1 = calculate_ng_rate_for_speed(df_machine1, speed_bins)
ng_rates_machine2 = calculate_ng_rate_for_speed(df_machine2, speed_bins)

# グラフの作成
fig, ax = plt.subplots(figsize=(15, 10))

bar_width = 0.0004  # バーの幅を調整

ax.bar(speed_bins[:-1], ng_rates_machine1, width=bar_width, alpha=0.5, color='blue', label='1号機')
ax.bar(speed_bins[:-1] + bar_width, ng_rates_machine2, width=bar_width, alpha=0.5, color='orange', label='2号機')

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

# グラフの保存
speed_ng_rate_filename = os.path.join(SPEED_NG_RATE_DIR, f'低速速度とNG率の関係_{CURRENT_TIME}.png')
plt.savefig(speed_ng_rate_filename)

print(f"低速（平均）速度とNG率の関係グラフを保存しました: {speed_ng_rate_filename}")

# グラフの表示（必要に応じてコメントアウトを解除）
# plt.show()

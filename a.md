import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'  # または 'IPAexGothic', 'Yu Gothic'などを試してみてください

# データの読み込み
df = pd.read_csv('casting_data.csv')

# データの前処理
date_columns = ['日時', '出荷検査日時', '加工検査日時']
for col in date_columns:
    df[col] = pd.to_datetime(df[col])

# 時系列順にソート
df = df.sort_values('日時')

# データの概要を確認
print(df.info())
print(df.describe())

# 鋳造条件の列名を取得（int型とfloat型の列）
casting_condition_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
casting_condition_columns = [col for col in casting_condition_columns if col != '目的変数']

# 出力ディレクトリの作成
output_dir_timevis = r'..\data\output\time_vis\time_series'
os.makedirs(output_dir_timevis, exist_ok=True)

# 日ごとのデータを分割する関数
def split_days(df):
    return [group for _, group in df.groupby(df['日時'].dt.date)]

# Y軸の範囲を計算する関数
def get_y_range(machine_df, condition):
    y_min = machine_df[condition].min()
    y_max = machine_df[condition].max()
    y_range = y_max - y_min
    return y_min - 0.1 * y_range, y_max + 0.3 * y_range  # マークのスペースを確保

# 時系列での鋳造条件の変化を可視化（日ごと、鋳造機名ごと、品番ごと）
days = split_days(df)

for condition in casting_condition_columns:
    y_ranges = {machine: get_y_range(df[df['鋳造機名'] == machine], condition) for machine in df['鋳造機名'].unique()}
    
    for machine in df['鋳造機名'].unique():
        y_min, y_max = y_ranges[machine]
        mark_y = y_min + 0.95 * (y_max - y_min)  # マークの位置を設定
        
        pdf_filename = os.path.join(output_dir_timevis, f'timevis_{condition}_{machine}_{datetime.now().strftime("%y%m%d%H%M")}.pdf')
        with PdfPages(pdf_filename) as pdf:
            for day_data in days:
                day_df = day_data[day_data['鋳造機名'] == machine]
                
                if day_df.empty:
                    continue
                
                fig, ax = plt.subplots(figsize=(20, 10))
                
                # 品番ごとに異なるマーカーを使用（4種類に限定）
                unique_products = day_df['品番'].unique()
                markers = ['o', 's', '^', 'D']
                colors = ['blue', 'green', 'orange', 'purple']
                
                for i, product in enumerate(unique_products):
                    product_df = day_df[day_df['品番'] == product]
                    ax.scatter(product_df['日時'], product_df[condition], 
                               label=f'品番: {product}', 
                               marker=markers[i % len(markers)],
                               color=colors[i % len(colors)],
                               s=50, zorder=3)  # マーカーサイズを調整
                    
                    # 5分以内のデータ点を線で結ぶ
                    product_df = product_df.sort_values('日時')
                    for j in range(1, len(product_df)):
                        if (product_df['日時'].iloc[j] - product_df['日時'].iloc[j-1]).total_seconds() <= 300:
                            ax.plot(product_df['日時'].iloc[j-1:j+1], product_df[condition].iloc[j-1:j+1], 
                                    color=colors[i % len(colors)], alpha=0.5, zorder=2)
                
                # NGのマーク
                ng_mask = day_df['目的変数'] == 1
                ax.scatter(day_df[ng_mask]['日時'], [mark_y]*sum(ng_mask), color='red', marker='x', s=200, label='NG', zorder=5)
                
                # NGの出荷検査と加工検査のマーク
                for _, row in day_df[ng_mask].iterrows():
                    ax.annotate(row['日時'].strftime('%H:%M'), xy=(row['日時'], mark_y), xytext=(0, 10), 
                                textcoords='offset points', ha='center', va='bottom', zorder=6)
                    ax.annotate('', xy=(row['出荷検査日時'], mark_y), xytext=(row['日時'], mark_y),
                                arrowprops=dict(arrowstyle='->', color='blue', lw=2), zorder=4)
                    ax.annotate('', xy=(row['加工検査日時'], mark_y), xytext=(row['日時'], mark_y),
                                arrowprops=dict(arrowstyle='->', color='purple', lw=2), zorder=4)
                
                ax.scatter(day_df[ng_mask]['出荷検査日時'], [mark_y]*sum(ng_mask), color='blue', marker='s', s=200, label='出荷検査(NG)', zorder=5)
                ax.scatter(day_df[ng_mask]['加工検査日時'], [mark_y]*sum(ng_mask), color='purple', marker='^', s=200, label='加工検査(NG)', zorder=5)
                
                ax.set_xlabel('時間')
                ax.set_ylabel(f'{condition}の値')
                date = day_df['日時'].iloc[0].date()
                ax.set_title(f'{condition}の時間経過による変化 (鋳造機名: {machine})\n{date.strftime("%Y/%m/%d")}')
                ax.set_xlim(day_df['日時'].min(), day_df['日時'].max())
                ax.set_ylim(y_min, y_max)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                # plt.show()  # コメントアウトを外すと表示されます

            # 加工検査日時がある日のグラフも表示
            inspection_dates = df[(df['鋳造機名'] == machine) & (df['加工検査日時'].notnull())]['加工検査日時'].dt.date.unique()
            for inspection_date in inspection_dates:
                day_df = df[(df['鋳造機名'] == machine) & (df['加工検査日時'].dt.date == inspection_date)]
                
                fig, ax = plt.subplots(figsize=(20, 10))
                
                for i, product in enumerate(unique_products):
                    product_df = day_df[day_df['品番'] == product]
                    ax.scatter(product_df['加工検査日時'], product_df[condition], 
                               label=f'品番: {product}', 
                               marker=markers[i % len(markers)],
                               color=colors[i % len(colors)],
                               s=50, zorder=3)
                
                # NGのマーク
                ng_mask = day_df['目的変数'] == 1
                ax.scatter(day_df[ng_mask]['加工検査日時'], [mark_y]*sum(ng_mask), color='red', marker='x', s=200, label='NG', zorder=5)
                
                # 加工検査日時のマーク
                ax.scatter(day_df['加工検査日時'], [mark_y]*len(day_df), color='purple', marker='^', s=200, label='加工検査', zorder=5)
                
                for _, row in day_df.iterrows():
                    ax.annotate(row['日時'].strftime('%H:%M'), xy=(row['加工検査日時'], mark_y), xytext=(0, 10), 
                                textcoords='offset points', ha='center', va='bottom', zorder=6)
                
                ax.set_xlabel('時間')
                ax.set_ylabel(f'{condition}の値')
                ax.set_title(f'加工検査日の{condition}の変化 (鋳造機名: {machine})\n{inspection_date.strftime("%Y/%m/%d")}')
                ax.set_xlim(day_df['加工検査日時'].min(), day_df['加工検査日時'].max())
                ax.set_ylim(y_min, y_max)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                # plt.show()  # コメントアウトを外すと表示されます

print("全ての鋳造条件のグラフと加工検査日のグラフがPDFとして保存されました。")
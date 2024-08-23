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
output_dir_ng = r'..\data\output\time_vis\ng_analysis'
os.makedirs(output_dir_timevis, exist_ok=True)
os.makedirs(output_dir_ng, exist_ok=True)

# 日ごとのデータを分割する関数
def split_days(df):
    return [group for _, group in df.groupby(df['日時'].dt.date)]

# 時系列での鋳造条件の変化を可視化（日ごと、鋳造機名ごと、品番ごと）
days = split_days(df)

for condition in casting_condition_columns:
    for machine in df['鋳造機名'].unique():
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
                               s=50)  # マーカーサイズを調整
                    
                    # 5分以内のデータ点を線で結ぶ
                    product_df = product_df.sort_values('日時')
                    for j in range(1, len(product_df)):
                        if (product_df['日時'].iloc[j] - product_df['日時'].iloc[j-1]).total_seconds() <= 300:
                            ax.plot(product_df['日時'].iloc[j-1:j+1], product_df[condition].iloc[j-1:j+1], 
                                    color=colors[i % len(colors)], alpha=0.5)
                
                # NGのマーク
                ng_mask = day_df['目的変数'] == 1
                ax.scatter(day_df[ng_mask]['日時'], [ax.get_ylim()[1]]*sum(ng_mask), color='red', marker='x', s=200, label='NG')
                
                # NGの出荷検査と加工検査のマーク
                for _, row in day_df[ng_mask].iterrows():
                    ax.annotate(row['日時'].strftime('%H:%M'), xy=(row['出荷検査日時'], ax.get_ylim()[1]), xytext=(0, 10), 
                                textcoords='offset points', ha='center', va='bottom')
                    ax.annotate('', xy=(row['出荷検査日時'], ax.get_ylim()[1]), xytext=(row['日時'], ax.get_ylim()[1]),
                                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
                    ax.annotate(row['日時'].strftime('%H:%M'), xy=(row['加工検査日時'], ax.get_ylim()[1]), xytext=(0, 10), 
                                textcoords='offset points', ha='center', va='bottom')
                    ax.annotate('', xy=(row['加工検査日時'], ax.get_ylim()[1]), xytext=(row['日時'], ax.get_ylim()[1]),
                                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
                
                ax.scatter(day_df[ng_mask]['出荷検査日時'], [ax.get_ylim()[1]]*sum(ng_mask), color='blue', marker='s', s=200, label='出荷検査(NG)')
                ax.scatter(day_df[ng_mask]['加工検査日時'], [ax.get_ylim()[1]]*sum(ng_mask), color='purple', marker='^', s=200, label='加工検査(NG)')
                
                ax.set_xlabel('時間')
                ax.set_ylabel(f'{condition}の値')
                date = day_df['日時'].iloc[0].date()
                ax.set_title(f'{condition}の時間経過による変化 (鋳造機名: {machine})\n{date.strftime("%Y/%m/%d")}')
                ax.set_xlim(day_df['日時'].min(), day_df['日時'].max())
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                # plt.show()  # コメントアウトを外すと表示されます

print("全ての鋳造条件のグラフがPDFとして保存されました。")

# NGが発生した際の前後のデータを分析
for machine in df['鋳造機名'].unique():
    machine_df = df[df['鋳造機名'] == machine].sort_values('日時')
    ng_indices = machine_df[machine_df['目的変数'] == 1].index
    
    pdf_filename = os.path.join(output_dir_ng, f'timevis_ng_analysis_{machine}_{datetime.now().strftime("%y%m%d%H%M")}.pdf')
    with PdfPages(pdf_filename) as pdf:
        for ng_index in ng_indices:
            ng_time = machine_df.loc[ng_index, '日時']
            start_time = ng_time - timedelta(minutes=30)
            end_time = ng_time + timedelta(minutes=30)
            
            ng_period = machine_df[(machine_df['日時'] >= start_time) & (machine_df['日時'] <= end_time)]
            
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # 品番を取得
            product = ng_period['品番'].iloc[0]
            
            for condition in casting_condition_columns:
                ax.scatter(ng_period['日時'], ng_period[condition], label=condition, s=50)
                
                # 5分以内のデータ点を線で結ぶ
                for j in range(1, len(ng_period)):
                    if (ng_period['日時'].iloc[j] - ng_period['日時'].iloc[j-1]).total_seconds() <= 300:
                        ax.plot(ng_period['日時'].iloc[j-1:j+1], ng_period[condition].iloc[j-1:j+1], alpha=0.5)
            
            ax.axvline(x=ng_time, color='r', linestyle='--', label='NG発生')
            
            # NGのマーク
            ng_mask = ng_period['目的変数'] == 1
            ax.scatter(ng_period[ng_mask]['日時'], [ax.get_ylim()[1]]*sum(ng_mask), color='red', marker='x', s=200, label='NG')
            
            # NGの出荷検査と加工検査のマーク
            for _, row in ng_period[ng_mask].iterrows():
                ax.annotate(row['日時'].strftime('%H:%M'), xy=(row['出荷検査日時'], ax.get_ylim()[1]), xytext=(0, 10), 
                            textcoords='offset points', ha='center', va='bottom')
                ax.annotate('', xy=(row['出荷検査日時'], ax.get_ylim()[1]), xytext=(row['日時'], ax.get_ylim()[1]),
                            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
                ax.annotate(row['日時'].strftime('%H:%M'), xy=(row['加工検査日時'], ax.get_ylim()[1]), xytext=(0, 10), 
                            textcoords='offset points', ha='center', va='bottom')
                ax.annotate('', xy=(row['加工検査日時'], ax.get_ylim()[1]), xytext=(row['日時'], ax.get_ylim()[1]),
                            arrowprops=dict(arrowstyle='->', color='purple', lw=2))
            
            ax.scatter(ng_period[ng_mask]['出荷検査日時'], [ax.get_ylim()[1]]*sum(ng_mask), color='blue', marker='s', s=200, label='出荷検査(NG)')
            ax.scatter(ng_period[ng_mask]['加工検査日時'], [ax.get_ylim()[1]]*sum(ng_mask), color='purple', marker='^', s=200, label='加工検査(NG)')
            
            ax.set_xlabel('時間')
            ax.set_ylabel('鋳造条件の値')
            ax.set_title(f'NG発生前後の鋳造条件の変化 (鋳造機名: {machine}, 品番: {product})\n{start_time.strftime("%Y/%m/%d %H:%M")} ~ {end_time.strftime("%Y/%m/%d %H:%M")}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            # plt.show()  # コメントアウトを外すと表示されます

        # 加工検査日時がある日のグラフも表示
        inspection_dates = machine_df[machine_df['加工検査日時'].notnull()]['加工検査日時'].dt.date.unique()
        for inspection_date in inspection_dates:
            day_df = machine_df[machine_df['日時'].dt.date == inspection_date]
            
            fig, ax = plt.subplots(figsize=(20, 10))
            
            for condition in casting_condition_columns:
                ax.scatter(day_df['日時'], day_df[condition], label=condition, s=50)
                
                # 5分以内のデータ点を線で結ぶ
                for j in range(1, len(day_df)):
                    if (day_df['日時'].iloc[j] - day_df['日時'].iloc[j-1]).total_seconds() <= 300:
                        ax.plot(day_df['日時'].iloc[j-1:j+1], day_df[condition].iloc[j-1:j+1], alpha=0.5)
            
            # NGのマーク
            ng_mask = day_df['目的変数'] == 1
            ax.scatter(day_df[ng_mask]['日時'], [ax.get_ylim()[1]]*sum(ng_mask), color='red', marker='x', s=200, label='NG')
            
            # NGの出荷検査と加工検査のマーク
            for _, row in day_df[ng_mask].iterrows():
                ax.annotate(row['日時'].strftime('%H:%M'), xy=(row['出荷検査日時'], ax.get_ylim()[1]), xytext=(0, 10), 
                            textcoords='offset points', ha='center', va='bottom')
                ax.annotate('', xy=(row['出荷検査日時'], ax.get_ylim()[1]), xytext=(row['日時'], ax.get_ylim()[1]),
                            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
                ax.annotate(row['日時'].strftime('%H:%M'), xy=(row['加工検査日時'], ax.get_ylim()[1]), xytext=(0, 10), 
                            textcoords='offset points', ha='center', va='bottom')
                ax.annotate('', xy=(row['加工検査日時'], ax.get_ylim()[1]), xytext=(row['日時'], ax.get_ylim()[1]),
                            arrowprops=dict(arrowstyle='->', color='purple', lw=2))
            
            ax.scatter(day_df[ng_mask]['出荷検査日時'], [ax.get_ylim()[1]]*sum(ng_mask), color='blue', marker='s', s=200, label='出荷検査(NG)')
            ax.scatter(day_df[ng_mask]['加工検査日時'], [ax.get_ylim()[1]]*sum(ng_mask), color='purple', marker='^', s=200, label='加工検査(NG)')
            
            ax.set_xlabel('時間')
            ax.set_ylabel('鋳造条件の値')
            ax.set_title(f'加工検査日の鋳造条件の変化 (鋳造機名: {machine})\n{inspection_date.strftime("%Y/%m/%d")}')
            ax.set_xlim(day_df['日時'].min(), day_df['日時'].max())
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            # plt.show()  # コメントアウトを外すと表示されます

print("NGが発生した際の前後のデータ分析グラフ、および加工検査日のグラフがPDFとして保存されました。")

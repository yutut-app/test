import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from matplotlib.backends.backend_pdf import PdfPages

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
output_dir = r'..\data\output\time_vis'
os.makedirs(output_dir, exist_ok=True)

# 週ごとのデータを分割する関数
def split_weeks(df):
    weeks = []
    current_week = []
    current_monday = df['日時'].iloc[0].floor('D') - timedelta(days=df['日時'].iloc[0].weekday())
    
    for _, row in df.iterrows():
        if row['日時'].floor('D') >= current_monday + timedelta(days=7):
            weeks.append(current_week)
            current_week = []
            current_monday += timedelta(days=7)
        current_week.append(row)
    
    if current_week:
        weeks.append(current_week)
    
    return weeks

# 時系列での鋳造条件の変化を可視化（週ごと、鋳造機名ごと、品番ごと）
weeks = split_weeks(df)

for condition in casting_condition_columns:
    pdf_filename = os.path.join(output_dir, f'timevis_{condition}_{datetime.now().strftime("%y%m%d%H%M")}.pdf')
    with PdfPages(pdf_filename) as pdf:
        for week_num, week_data in enumerate(weeks):
            week_df = pd.DataFrame(week_data)
            
            # 鋳造機名ごとにグラフを分ける
            unique_machines = week_df['鋳造機名'].unique()
            for machine in unique_machines:
                machine_df = week_df[week_df['鋳造機名'] == machine]
                
                fig, ax = plt.subplots(figsize=(20, 10))
                
                # 品番ごとに線の種類を変える
                unique_products = machine_df['品番'].unique()
                line_styles = ['-', '--', '-.', ':']  # 線の種類のリスト
                
                for i, product in enumerate(unique_products):
                    product_df = machine_df[machine_df['品番'] == product]
                    ax.plot(product_df['日時'], product_df[condition], 
                            label=f'品番: {product}', 
                            linestyle=line_styles[i % len(line_styles)])
                
                # OK/NGのマーク
                for _, row in machine_df.iterrows():
                    if row['目的変数'] == 0:
                        ax.plot(row['日時'], ax.get_ylim()[1], 'go', markersize=10, label='OK' if 'OK' not in ax.get_legend_handles_labels()[1] else "")
                    else:
                        ax.plot(row['日時'], ax.get_ylim()[1], 'ro', markersize=10, label='NG' if 'NG' not in ax.get_legend_handles_labels()[1] else "")
                
                # 出荷検査と加工検査のマーク
                ax.scatter(machine_df['出荷検査日時'], [ax.get_ylim()[1]]*len(machine_df), color='blue', marker='s', s=100, label='出荷検査')
                ax.scatter(machine_df[machine_df['目的変数'] == 1]['加工検査日時'], [ax.get_ylim()[1]]*len(machine_df[machine_df['目的変数'] == 1]), color='purple', marker='^', s=100, label='加工検査')
                
                # 5分以上間隔が空いている箇所で線を切る
                for i in range(1, len(machine_df)):
                    if (machine_df['日時'].iloc[i] - machine_df['日時'].iloc[i-1]).total_seconds() >= 300:
                        ax.axvline(machine_df['日時'].iloc[i] - timedelta(seconds=150), color='gray', linestyle='--', alpha=0.5)
                
                ax.set_xlabel('時間')
                ax.set_ylabel(f'{condition}の値')
                start_date = week_df['日時'].iloc[0].floor('D')
                end_date = start_date + timedelta(days=6)
                ax.set_title(f'{condition}の時間経過による変化 (鋳造機名: {machine})\n{start_date.strftime("%Y/%m/%d")} ~ {end_date.strftime("%Y/%m/%d")}')
                ax.set_xlim(start_date, end_date + timedelta(days=1))
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                # plt.show()  # コメントアウトを外すと表示されます

print("全ての鋳造条件のグラフがPDFとして保存されました。")

# NGが発生した際の前後のデータを分析
pdf_filename = os.path.join(output_dir, f'timevis_ng_analysis_{datetime.now().strftime("%y%m%d%H%M")}.pdf')
with PdfPages(pdf_filename) as pdf:
    ng_indices = df[df['目的変数'] == 1].index
    for ng_index in ng_indices:
        start_index = max(0, ng_index - 5)
        end_index = min(len(df), ng_index + 6)
        ng_period = df.iloc[start_index:end_index]
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 鋳造機名と品番を取得
        machine = ng_period['鋳造機名'].iloc[0]
        product = ng_period['品番'].iloc[0]
        
        for condition in casting_condition_columns:
            ax.plot(ng_period['日時'], ng_period[condition], label=condition)
        
        ax.axvline(x=df.loc[ng_index, '日時'], color='r', linestyle='--', label='NG発生')
        
        # OK/NGのマーク
        for _, row in ng_period.iterrows():
            if row['目的変数'] == 0:
                ax.plot(row['日時'], ax.get_ylim()[1], 'go', markersize=10, label='OK' if 'OK' not in ax.get_legend_handles_labels()[1] else "")
            else:
                ax.plot(row['日時'], ax.get_ylim()[1], 'ro', markersize=10, label='NG' if 'NG' not in ax.get_legend_handles_labels()[1] else "")
        
        # 出荷検査と加工検査のマーク
        ax.scatter(ng_period['出荷検査日時'], [ax.get_ylim()[1]]*len(ng_period), color='blue', marker='s', s=100, label='出荷検査')
        ax.scatter(ng_period[ng_period['目的変数'] == 1]['加工検査日時'], [ax.get_ylim()[1]]*len(ng_period[ng_period['目的変数'] == 1]), color='purple', marker='^', s=100, label='加工検査')
        
        ax.set_xlabel('時間')
        ax.set_ylabel('鋳造条件の値')
        start_date = ng_period['日時'].iloc[0]
        end_date = ng_period['日時'].iloc[-1]
        ax.set_title(f'NG発生前後の鋳造条件の変化 (鋳造機名: {machine}, 品番: {product})\n{start_date.strftime("%Y/%m/%d %H:%M")} ~ {end_date.strftime("%Y/%m/%d %H:%M")}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        # plt.show()  # コメントアウトを外すと表示されます

print("NGが発生した際の前後のデータ分析グラフがPDFとして保存されました。")

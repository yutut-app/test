# 必要なライブラリをインポート
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# データの読み込み
# CSVファイルからデータを読み込みます。ファイルパスは適宜変更してください。
data_frame = pd.read_csv('../data/input/casting_data.csv')

# 日時列を日時型に変換
date_time_columns = ['日時', '出荷検査日時', '加工検査日時']
for column in date_time_columns:
    data_frame[column] = pd.to_datetime(data_frame[column])

# 鋳造条件の列を特定（int型とfloat型の列）
casting_condition_columns = data_frame.select_dtypes(include=['int64', 'float64']).columns.tolist()
casting_condition_columns.remove('目的変数')  # 目的変数は除外

# 時系列データの可視化関数
def visualize_time_series_data(data_frame, condition):
    plt.figure(figsize=(20, 10))
    
    # 1週間ごとにデータをグループ化
    data_frame['週'] = data_frame['日時'].dt.to_period('W-MON')  # 月曜日始まりの週に設定
    weeks = data_frame['週'].unique()
    
    for week_index, week in enumerate(weeks):
        week_data = data_frame[data_frame['週'] == week]
        
        # 月曜日から始まるようにデータを調整
        start_of_week = week.start_time
        
        # X軸: 鋳造条件の値、Y軸: 時刻
        plt.subplot(1, len(weeks), week_index + 1)
        
        previous_time = None
        for _, row in week_data.iterrows():
            current_time = row['日時']
            
            # 5分以上の間隔がある場合は線を繋げない
            if previous_time is not None and (current_time - previous_time) > timedelta(minutes=5):
                plt.plot(row[condition], current_time.time(), marker='o', color='none', markersize=5)
            else:
                color = 'green' if row['目的変数'] == 0 else 'red'
                plt.plot(row[condition], current_time.time(), marker='o', color=color, markersize=5)
            
            # 出荷検査と加工検査のマーク
            if pd.notna(row['出荷検査日時']):
                plt.plot(row[condition], row['出荷検査日時'].time(), marker='s', color='blue', markersize=3)
            if pd.notna(row['加工検査日時']) and row['目的変数'] == 1:
                plt.plot(row[condition], row['加工検査日時'].time(), marker='*', color='purple', markersize=3)
            
            previous_time = current_time
        
        plt.title(f'Week of {start_of_week.date()}')
        plt.xlabel(condition)
        plt.ylabel('Time')
    
    plt.suptitle(f'Time Series Visualization for {condition}')
    plt.tight_layout()
    
    # グラフをPDFとして保存
    current_time = datetime.now().strftime("%y%m%d%H%M")
    output_directory = '../data/output/time_vis'
    os.makedirs(output_directory, exist_ok=True)
    plt.savefig(f'{output_directory}/timevis_{condition}_{current_time}.pdf')
    plt.close()

# 各鋳造条件に対して可視化を実行
for condition in casting_condition_columns:
    visualize_time_series_data(data_frame, condition)
    print(f'Visualization for {condition} saved.')

# グラフを表示したい場合は、以下のコメントを外してください
# plt.show()

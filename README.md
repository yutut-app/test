# 必要なライブラリをインポートします
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# データの読み込み
# CSVファイルからデータを読み込みます。ファイル名は適宜変更してください。
df = pd.read_csv('casting_process_data.csv')

# データの前処理
# 日付列を日時型に変換します
date_columns = ['日時', '出荷検査日時', '加工検査日時']
for col in date_columns:
    df[col] = pd.to_datetime(df[col])

# データの確認
print(df.head())
print(df.info())

# 時系列グラフの作成
plt.figure(figsize=(15, 8))
sns.set_style("whitegrid")

# OKとNGのデータを分けて、それぞれ異なる色で表示します
ok_data = df[df['目的関数'] == 0]
ng_data = df[df['目的関数'] == 1]

# OKデータの折れ線グラフ
plt.plot(ok_data['日時'], ok_data['鋳造条件1'], color='blue', label='OK', marker='o')

# NGデータの折れ線グラフ
plt.plot(ng_data['日時'], ng_data['鋳造条件1'], color='red', label='NG', marker='x')

# 5分以上の間隔があるデータポイントの間は線を繋げないようにします
for i in range(1, len(df)):
    if (df['日時'].iloc[i] - df['日時'].iloc[i-1]) > timedelta(minutes=5):
        plt.axvline(x=df['日時'].iloc[i], color='gray', linestyle='--', alpha=0.5)

plt.title('鋳造条件1の時系列変化', fontsize=16)
plt.xlabel('日時', fontsize=12)
plt.ylabel('鋳造条件1の値', fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 鋳造機名ごとの分析
plt.figure(figsize=(15, 8))
for machine in df['鋳造機名'].unique():
    machine_data = df[df['鋳造機名'] == machine]
    plt.plot(machine_data['日時'], machine_data['鋳造条件1'], label=machine)

plt.title('鋳造機名ごとの鋳造条件1の時系列変化', fontsize=16)
plt.xlabel('日時', fontsize=12)
plt.ylabel('鋳造条件1の値', fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 品番（型番）ごとの分析
plt.figure(figsize=(15, 8))
for product in df['品番'].unique():
    product_data = df[df['品番'] == product]
    plt.plot(product_data['日時'], product_data['鋳造条件1'], label=product)

plt.title('品番ごとの鋳造条件1の時系列変化', fontsize=16)
plt.xlabel('日時', fontsize=12)
plt.ylabel('鋳造条件1の値', fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# OKとNGの割合の時系列変化
df['date'] = df['日時'].dt.date
daily_ratio = df.groupby('date')['目的関数'].mean()

plt.figure(figsize=(15, 8))
plt.plot(daily_ratio.index, daily_ratio.values, marker='o')
plt.title('日ごとのNG率の推移', fontsize=16)
plt.xlabel('日付', fontsize=12)
plt.ylabel('NG率', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

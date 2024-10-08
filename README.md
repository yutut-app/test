# 1. ライブラリのインポート

# まず、必要なライブラリをインストールします
# !pip install -r requirements.txt

# 次に、必要なライブラリをインポートします
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# 日本語フォントの設定
plt.rcParams['font.family'] = "Yu Gothic"

# 2. データの読み込み

# データファイルのパスを設定
defected_data_path = r"../data/output/defect_data"
defected_csv = "defects_data.csv"
defected_csv_path = os.path.join(defected_data_path, defected_csv)

# CSVファイルを読み込み、データフレームを作成
df = pd.read_csv(defected_csv_path)
print(f"データフレームの形状：{df.shape}")

# すべての列を表示するように設定
pd.set_option('display.max_columns', None)

# データフレームの先頭5行を表示
df.head()

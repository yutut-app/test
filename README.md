import pandas as pd
import numpy as np

# データの読み込み（すでに読み込んでいる場合は不要）
# df = pd.read_csv('path_to_your_data.csv')

# 現在のwork_idを基に新しいwork_idを作成する関数
def create_new_work_id(group):
    return np.repeat(np.arange(len(group) // 2), 2)

# データフレームを現在のwork_idでソート
df = df.sort_values('work_id')

# 新しいwork_idを作成
df['new_work_id'] = df.groupby('work_id').transform(create_new_work_id)

# work_idを更新
df['work_id'] = df.groupby('work_id')['new_work_id'].transform(lambda x: f"{x.iloc[0]:05d}")

# new_work_id列を削除（必要ない場合）
df = df.drop('new_work_id', axis=1)

# 結果の確認
print(df[['image_name', 'work_id']].head(10))

# ユニークなワーク数の確認
print(f"\nユニークなワーク数: {df['work_id'].nunique()}")

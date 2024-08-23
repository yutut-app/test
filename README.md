
```markdown
# 6. クリーニング済みデータの保存

データの整理（クリーニング）が終わったら、その結果を保存することが大切です。これは、きれいにしたデータを後で使えるようにするためです。

## なぜデータを保存するの？

1. 同じ分析を後でもう一度行える：
   - 保存したデータを使えば、同じ結果を得られます。
   - これは、自分の分析が正しいことを他の人に示すのに役立ちます。

2. 時間の節約：
   - データの整理には時間がかかります。
   - 保存しておけば、次に分析するときにまた最初からやり直す必要がありません。

3. データの記録：
   - いつデータを整理したかが分かるように、ファイル名に日付をつけます。
   - これで、古いデータと新しいデータを区別できます。

4. 他の人と共有しやすい：
   - 整理したデータを保存しておけば、チームの他のメンバーにも簡単に渡せます。

## データの保存方法？

データは「CSV」という形式で保存します。CSVは多くのプログラムで読み込めるので、とても便利です。

以下のコードでは：
1. 今日の日付を取得します。
2. データを保存する場所（フォルダ）を決めます。
3. ファイルの名前を「data_cleansing_年月日.csv」という形で作ります。
4. 整理したデータをその名前で保存します。

このようにすることで、いつ整理したデータなのかが一目で分かり、後で必要なときにすぐに使えるようになります。
```

```python
import os
from datetime import datetime

# 今日の日付を取得します
current_date = datetime.now().strftime("%y%m%d")

# データを保存する場所を決めます
current_dir = os.getcwd()  # 今いる場所
output_data_path = os.path.join(current_dir, "data", "output")  # 保存先のフォルダ

# 保存先のフォルダがなければ作ります
if not os.path.exists(output_data_path):
    os.makedirs(output_data_path)

# ファイルの名前を決めます
filename = f"data_cleansing_{current_date}.csv"

# 保存する場所とファイル名を組み合わせます
full_path = os.path.join(output_data_path, filename)

# 整理したデータを保存します
df_analysis_cleaned.to_csv(full_path, index=False)

print(f"整理したデータをここに保存しました：\n{full_path}")
```

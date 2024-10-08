ご要望を理解しました。以下に、EDAを行うためのJupyter Notebookのコードを、Markdownとコードセルを区別して生成します。

```markdown
# 欠陥データのEDAと性能評価

## 1. ライブラリのインポート
```

```python
# 必要なライブラリをインストール
%pip install -r requirements.txt

# ライブラリをインポート
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# プロットの日本語フォント設定
plt.rcParams['font.family'] = "Yu Gothic"
```

```markdown
## 2. データの読み込み
```

```python
# データファイルのパスを設定
defected_data_path = r"../data/output/defect_data"
defected_csv = "defects_data.csv"
defected_csv_path = os.path.join(defected_data_path, defected_csv)

# CSVファイルを読み込む
df = pd.read_csv(defected_csv_path)
print(f"データフレームの形状：{df.shape}")

# 全列を表示するように設定
pd.set_option('display.max_columns', None)

# データフレームの先頭を表示
df.head()
```

このコードは、必要なライブラリをインポートし、CSVファイルからデータを読み込みます。また、データフレームの形状を表示し、先頭の数行を表示します。

次のステップでは、このデータを使用してEDA（探索的データ分析）を行うことができます。例えば、以下のような分析が考えられます：

1. データの基本統計量の確認
2. 欠損値の確認と処理
3. 特徴量の分布の可視化
4. 特徴量間の相関分析
5. 目的変数（defect_label）と各特徴量の関係の可視化

これらの分析を進めるにあたって、何か具体的に焦点を当てたい部分はありますか？

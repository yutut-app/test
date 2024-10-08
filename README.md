承知しました。EDAを行い、目的変数と独立変数の関係を可視化するコードを生成します。stripplotを使用し、各特徴量ごとにグラフを作成し、PDFとして保存します。

```markdown
## 5. 探索的データ分析（EDA）

ここでは、目的変数（defect_label）と各独立変数の関係を可視化します。stripplotを使用して、各特徴量ごとにグラフを作成し、PDFファイルとして保存します。この分析により、どの特徴量が欠陥の有無と強い関連性を持つかを視覚的に確認できます。
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime
import numpy as np

# 出力ディレクトリの設定
defected_data_path = r"../data/output/defect_data"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
pdf_filename = os.path.join(defected_data_path, f'eda_defect_analysis_{current_time}.pdf')

# 独立変数のリスト
independent_vars = ['width', 'height', 'area', 'perimeter', 'eccentricity', 'orientation', 
                    'major_axis_length', 'minor_axis_length', 'solidity', 'extent', 
                    'aspect_ratio', 'max_length']

# 改良されたサンプリング関数
def sample_data(df, n=10000):
    ok_data = df[df['defect_label'] == 0]
    ng_data = df[df['defect_label'] == 1]
    
    if len(df) <= n:
        return df
    
    # OKデータは全て含める
    sampled_ok = ok_data
    
    # 残りのサンプル数をNGデータから抽出
    n_ng = n - len(sampled_ok)
    sampled_ng = ng_data.sample(n=min(n_ng, len(ng_data)), random_state=42)
    
    # OKデータとサンプリングしたNGデータを結合
    sampled_df = pd.concat([sampled_ok, sampled_ng]).sample(frac=1, random_state=42)
    
    return sampled_df

# グラフを表示するかどうかのフラグ
show_plots = False  # Trueにするとグラフを表示、Falseにすると表示しない

# 全データのサンプリング（一度だけ実行）
df_sampled = sample_data(df)

# PDFファイルを作成
with PdfPages(pdf_filename) as pdf:
    for var in independent_vars:
        # プロットの作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # catplotを使用してプロット
        sns.catplot(data=df_sampled, x=var, y='defect_label', kind='strip', 
                    jitter=True, alpha=0.5, height=6, aspect=2)
        
        # タイトルと軸ラベルの設定（日本語）
        plt.title(f'{var}と欠陥ラベルの関係', fontsize=16)
        plt.xlabel(var, fontsize=14)
        plt.ylabel('欠陥ラベル', fontsize=14)
        
        # x軸の目盛り数を調整
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        # 凡例の設定
        plt.legend(title='欠陥ラベル', labels=['正常 (0)', '欠陥 (1)'])
        
        # グラフの調整
        plt.tight_layout()
        
        # PDFに追加
        pdf.savefig()
        
        # グラフを表示（フラグがTrueの場合）
        if show_plots:
            plt.show()
        else:
            plt.close()

print(f"EDAグラフをPDFに保存しました: {pdf_filename}")
```

```markdown
このEDA（探索的データ分析）により、以下の情報が視覚化されました：

1. 各独立変数と欠陥ラベル（defect_label）の関係
2. 欠陥がある（NG）サンプルと欠陥がない（OK）サンプルの分布の違い

生成されたPDFファイルには、各独立変数に対するstripplotが含まれています。これらのグラフから以下のような情報を読み取ることができます：

- どの特徴量が欠陥の有無と強い関連性を持つか
- 欠陥がある場合とない場合で、各特徴量の分布にどのような違いがあるか
- 特徴量の中に外れ値や特異な分布を示すものはないか

これらの情報は、後続のモデリングや特徴量選択において非常に有用です。例えば：

- 欠陥の有無との相関が強い特徴量を重点的に使用する
- 分布に大きな違いがある特徴量に注目する
- 外れ値が多い特徴量の取り扱いを検討する

次のステップでは、これらの視覚的な分析結果を踏まえて、特徴量の選択や前処理、モデリング戦略の検討を行うことができます。
```

このコードを実行すると、指定されたパスにPDFファイルが生成されます。PDFには各独立変数とdefect_labelの関係を示すstripplotが含まれています。

次に進めるにあたって、これらのグラフから特に注目すべき点や、さらに詳しく分析したい特徴量はありますか？また、この視覚化の結果を踏まえて、どのような特徴量選択や前処理を行いたいかについてのアイデアはありますか？

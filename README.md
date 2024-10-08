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

# 出力ディレクトリの設定
defected_data_path = r"../data/output/defect_data"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
pdf_filename = os.path.join(defected_data_path, f'eda_defect_analysis_{current_time}.pdf')

# 独立変数のリスト
independent_vars = ['width', 'height', 'area', 'perimeter', 'eccentricity', 'orientation', 
                    'major_axis_length', 'minor_axis_length', 'solidity', 'extent', 
                    'aspect_ratio', 'max_length']

# グラフを表示するかどうかのフラグ
show_plots = False  # Trueにするとグラフを表示、Falseにすると表示しない

# PDFファイルを作成
with PdfPages(pdf_filename) as pdf:
    for var in independent_vars:
        # プロットの作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # OKとNGのデータを分離
        df_ok = df[df['defect_label'] == 0]
        df_ng = df[df['defect_label'] == 1]
        
        # OKのデータをプロット（透明度を下げる）
        sns.stripplot(data=df_ok, x=var, y='defect_label', color='blue', alpha=0.3, 
                      jitter=True, size=5, ax=ax)
        
        # NGのデータをプロット（前面に、大きく、透明度を上げる）
        sns.stripplot(data=df_ng, x=var, y='defect_label', color='red', alpha=1.0, 
                      jitter=True, size=10, ax=ax)
        
        # タイトルと軸ラベルの設定
        plt.title(f'Relationship between {var} and Defect Label')
        plt.xlabel(var)
        plt.ylabel('Defect Label')
        
        # 凡例の設定
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='OK (0)', 
                                  markerfacecolor='blue', markersize=10, alpha=0.3),
                           Line2D([0], [0], marker='o', color='w', label='NG (1)', 
                                  markerfacecolor='red', markersize=15)]
        ax.legend(handles=legend_elements, title='Defect Label')
        
        # グラフの調整
        plt.tight_layout()
        
        # PDFに追加
        pdf.savefig(fig)
        
        # グラフを表示（フラグがTrueの場合）
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

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

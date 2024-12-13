# 2. データの読み込み

本セクションでは、分類処理に必要なデータの読み込みについて説明する。

## 前提条件

分類処理を実行する前に、以下のデータが適切に配置されている必要がある：

1. 欠陥データの配置
   ```
   data/
   ├── output/
   │   └── defect_data/
   │       ├── defect_images/  # No1_Detecting_defect_candidates.ipynbの実行結果
   │       │   └── ... 
   │       └── defects_data.csv  # 特徴量データ
   ├── input/
   │   └── defects_template_dir/  # 手動で切り出した鋳巣画像
   │       ├── defect1.jpg
   │       ├── defect2.jpg
   │       └── ...
   ```

2. 重要な注意点
   - defects_template_dirには、手動で切り出した鋳巣（欠陥）の画像を配置する必要がある
   - これらの画像は教師データとして使用される
   - 鮮明で典型的な欠陥画像を選択することが推奨される

## パスの設定

1. データ格納場所の定義
   - `defected_data_path`：欠陥データのルートディレクトリ
   - `defected_image_path`：欠陥画像の保存ディレクトリ
   - `defects_template_path`：教師データ用の欠陥テンプレート格納ディレクトリ

2. CSVファイルの設定
   - `defected_csv`：特徴量データのファイル名
   - `defected_csv_path`：特徴量データの完全パス

## データの読み込み

1. CSVファイルの読み込み
   - pandas.read_csvを使用
   - 欠陥の特徴量データを取得
   - データ数の確認を出力

2. 読み込まれるデータの内容
   - 画像情報（ファイル名、パス）
   - 欠陥情報（位置、サイズ、特徴量）
   - ラベル情報（画像ラベル、欠陥ラベル）

これらのデータは後続の前処理および分類処理で使用される。特に、手動で切り出した鋳巣画像は、テンプレートマッチングの基準として重要な役割を果たす。



# 2. データの読み込み
# パスの設定 
defected_data_path = r"../data/output/defect_data"
defected_image_path = r"../data/output/defect_data/defect_images" 
defects_template_path = r"../data/input/defects_template_dir"
defected_csv = "defects_data.csv"
defected_csv_path = os.path.join(defected_data_path, defected_csv)

# CSVファイル読み込み
print("=== データの読み込み ===")
df = pd.read_csv(defected_csv_path)
print(f"読み込んだデータ数: {len(df)}")

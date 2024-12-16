以下に、修正した関数とそれに対応するmarkdownを記載します。

# 3. データの読み込み

## 関数の修正

```python
def load_shape_images(directory):
    """
    指定ディレクトリからShape1画像とNormal画像を読み込みます
    
    引数:
        directory (str): 画像が格納されているディレクトリのパス
    
    戻り値:
        list: (画像パス, 画像ファイル名)のタプルのリスト
    """
    shape_images = []
    # 一時的に画像を格納する辞書
    image_dict = {}
    
    # 指定ディレクトリ内の全ファイルを走査
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Shape1とNormal画像を検出
            if ("Shape1" in file or "Normal" in file) and file.endswith(".jpg"):
                # ファイル名からShape1/Normalの部分を除いた共通部分を取得
                base_name = file.replace("Shape1", "").replace("Normal", "")
                
                if base_name not in image_dict:
                    image_dict[base_name] = []
                image_dict[base_name].append((os.path.join(root, file), file))
    
    # 同一ワークの画像をまとめてリストに追加
    for base_name, images in image_dict.items():
        shape_images.extend(images)
    
    return shape_images
```

## markdown説明

### load_shape_images()の改修内容

本関数は、指定されたディレクトリから画像を読み込む機能を担う。以下の変更を実装：

1. 対象画像の拡張
   - `Shape1`画像：キーエンス処理後の画像
   - `Normal`画像：キーエンス処理前の撮影画像
   - どちらも同一ワークの異なる処理段階を示す

2. 画像のペア管理
   - ファイル名の共通部分を用いて同一ワークの画像を特定
   - 例：
     ```
     ABC_Shape1_XYZ.jpg
     ABC_Normal_XYZ.jpg
     ```
     この場合、`ABC_XYZ.jpg`が共通部分

3. データ構造
   - 中間処理：辞書形式で同一ワークの画像を一時的に管理
   - 最終出力：従来通りのリスト形式を維持
     - [(画像パス1, ファイル名1), (画像パス2, ファイル名2), ...]

4. 処理フロー
   - ディレクトリの再帰的走査
   - Shape1/Normal画像の識別
   - 同一ワーク画像のグループ化
   - 結果のリスト化

この改修により、キーエンス処理前後の画像を対応付けて管理することが可能となる。後続の処理でこれらの画像を比較分析する際に活用できる。

エラーの原因は、`load_shape_images()`から返されるデータ形式と`process_images()`が期待するデータ形式が一致していないためです。修正したコードを示します：

```python
def load_shape_images(directory):
    """
    指定ディレクトリからShape1画像とNormal画像を読み込みます
    
    引数:
        directory (str): 画像が格納されているディレクトリのパス
    
    戻り値:
        list: (Normal画像パス, Shape1画像パス, ファイル名)のタプルのリスト
    """
    image_pairs = []
    # 一時的に画像を格納する辞書
    image_dict = {}
    
    # 指定ディレクトリ内の全ファイルを走査
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                # Shape1とNormal画像を検出
                if "Shape1" in file or "Normal" in file:
                    # ファイル名からShape1/Normalの部分を除いた共通部分を取得
                    base_name = file.replace("Shape1", "").replace("Normal", "")
                    file_path = os.path.join(root, file)
                    
                    if base_name not in image_dict:
                        image_dict[base_name] = {'Normal': None, 'Shape1': None, 'filename': base_name}
                    
                    if "Normal" in file:
                        image_dict[base_name]['Normal'] = file_path
                    elif "Shape1" in file:
                        image_dict[base_name]['Shape1'] = file_path
    
    # ペアの画像が揃っているものだけをリストに追加
    for base_name, paths in image_dict.items():
        if paths['Normal'] is not None and paths['Shape1'] is not None:
            image_pairs.append((paths['Normal'], paths['Shape1'], paths['filename']))
    
    return image_pairs
```

主な変更点：
1. 戻り値の形式を`(Normal画像パス, Shape1画像パス, ファイル名)`のタプルに変更
2. 画像のペアを管理する辞書構造を改善
3. NormalとShape1の画像ペアが揃っているものだけを返すように変更

これにより、`process_images()`関数が期待する3つの値（Normal画像パス、Shape1画像パス、ファイル名）が正しく渡されるようになります。

使用例：
```python
# 画像の処理実行
processed_ng_images = process_images(ng_images, judge_templates, mask_templates)
#processed_ok_images = process_images(ok_images, judge_templates, mask_templates)
```

この修正により、"not enough values to unpack" エラーは解消されるはずです。

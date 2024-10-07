承知しました。元の画像ファイル名を保持するために、コードを修正する必要があります。この変更を行うには、画像の読み込みから最終的なデータ保存までの処理を通して、元のファイル名を追跡する必要があります。以下に、変更が必要な箇所を示します。

まず、`load_origin_keyence_images` 関数を修正して、元のファイル名を保持するようにします：

```python
def load_origin_keyence_images(directory):
    normal_images = {}
    shape_images = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "Normal" in file and file.endswith(".jpg"):
                base_name = file.replace("Normal", "")
                normal_images[base_name] = (os.path.join(root, file), file)  # ファイル名も保存
            elif "Shape" in file and file.endswith(".jpg"):
                base_name = file.replace("Shape", "")
                shape_images[base_name] = (os.path.join(root, file), file)  # ファイル名も保存
    
    matched_images = []
    for base_name in normal_images:
        if base_name in shape_images:
            matched_images.append((normal_images[base_name][0], shape_images[base_name][0], shape_images[base_name][1]))  # Shape画像のファイル名を追加
    return matched_images
```

次に、`process_images` 関数を修正して、元のファイル名を渡すようにします：

```python
def process_images(image_pairs):
    updated_images = []
    for origin_image_path, keyence_image_path, original_filename in image_pairs:
        cropped_image, cropped_keyence_image = remove_joint_part(origin_image_path, keyence_image_path)
        updated_images.append((cropped_image, cropped_keyence_image, original_filename))
    return updated_images
```


これらの変更により、`defect_data` の `'image_name'` フィールドには元の "Shape" 画像のファイル名が保存されるようになります。また、出力ディレクトリ構造も元のファイル名に基づいて作成されるように変更されています。
```python
import os
import pandas as pd

# ... (前のコードは変更なし) ...

# CSVファイルに出力する部分を以下のように変更
csv_output_dir = os.path.join(output_data_dir, "defect_data")
os.makedirs(csv_output_dir, exist_ok=True)
csv_output_path = os.path.join(csv_output_dir, "defects_data.csv")

df = pd.DataFrame(all_defects_data)

if os.path.exists(csv_output_path):
    # 既存のCSVファイルが存在する場合、列名なしで上書き
    df.to_csv(csv_output_path, mode='a', header=False, index=False)
    print(f"Appended defects data to existing file: {csv_output_path}")
else:
    # 新規作成の場合、列名ありで保存
    df.to_csv(csv_output_path, index=False)
    print(f"Created new defects data file: {csv_output_path}")

print(f"Defects data saved to {csv_output_path}")
```

これらの変更により、元のファイル名（`original_filename`）が処理の全段階を通して保持され、最終的に `defect_data` の `'image_name'` フィールドに保存されます。また、出力ディレクトリ構造も元のファイル名に基づいて作成されるように変更されています。

これらの変更を適用することで、元の "Shape" 画像のファイル名が正しく保持され、最終的なCSVファイルに記録されるようになります。

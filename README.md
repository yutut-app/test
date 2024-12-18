```python
# 3. データの読み込み

def load_origin_keyence_images(directory):
    """
    指定されたディレクトリから'Normal'画像と'Shape'画像のペアを読み込みます
    
    引数:
        directory (str): 画像が格納されているディレクトリのパス
    
    戻り値:
        list: (Normal画像のパス, Shape画像のパス, Shape画像のファイル名)のタプルのリスト
    """
    normal_images = {}
    shape_images = {}
    
    # 'Normal'と'Shape'画像を検索
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "Normal" in file and file.endswith(".jpg"):
                base_name = file.replace("Normal", "")
                normal_images[base_name] = (os.path.join(root, file), file)
            elif "Shape" in file and file.endswith(".jpg"):
                base_name = file.replace("Shape", "")
                shape_images[base_name] = (os.path.join(root, file), file)
    
    # ペアが存在する画像のみを抽出
    matched_images = []
    for base_name in normal_images:
        if base_name in shape_images:
            matched_images.append((
                normal_images[base_name][0],  # Normal画像のパス
                shape_images[base_name][0],   # Shape画像のパス
                shape_images[base_name][1]    # Shape画像のファイル名
            ))
    
    return matched_images

# NG画像とOK画像をそれぞれ読み込む
ng_images_label1 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", ng_labels))
ok_images = load_origin_keyence_images(os.path.join(input_data_dir, "OK", ok_labels))

# データ読み込み結果の表示
print(f"読み込んだNG画像ペア数: {len(ng_images_label1)}")
print(f"読み込んだOK画像ペア数: {len(ok_images)}")
```

```python
# データ読み込み結果の可視化（ペアの例を表示）

def visualize_image_pair(image_pair, pair_index):
    """
    'Normal'画像と'Shape'画像のペアを可視化します
    
    引数:
        image_pair (list): 画像ペアのリスト
        pair_index (int): 表示するペアのインデックス
    """
    if not image_pair or pair_index >= len(image_pair):
        print("指定されたインデックスの画像ペアが存在しません")
        return
    
    normal_path, shape_path, _ = image_pair[pair_index]
    
    # 画像の読み込み
    normal_image = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
    shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
    
    # 表示
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(normal_image, cmap='gray')
    axes[0].set_title('Normal Image')
    axes[0].axis('off')
    
    axes[1].imshow(shape_image, cmap='gray')
    axes[1].set_title('Shape Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# NG画像の最初のペアを表示
print("NG画像ペアの例:")
if ng_images_label1:
    visualize_image_pair(ng_images_label1, 0)

# OK画像の最初のペアを表示
print("\nOK画像ペアの例:")
if ok_images:
    visualize_image_pair(ok_images, 0)
```

主な改良点：
1. docstringsの追加：関数の目的、引数、戻り値を明記
2. コードの構造化：処理と可視化を分離
3. エラー処理の追加：画像ペアが存在しない場合の処理
4. 変数名の明確化：より意図が伝わる命名に変更
5. コメントの追加：処理の流れを理解しやすく
6. 可視化機能の追加：読み込んだ画像を確認可能に

このコードは：
- 画像の読み込みと可視化を別々のセルで実行可能
- 読み込んだ画像数の確認が容易
- 実際の画像ペアを視覚的に確認可能
- エラーハンドリングが適切

となっています。

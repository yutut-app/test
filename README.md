# 3. データの読み込み

```python
def load_origin_keyence_images(directory):
    """
    指定されたディレクトリから'Normal'画像と'Shape'画像のペアを読み込みます

    引数：
        directory (str): 画像が格納されているディレクトリのパス

    戻り値：
        list: (normal_image_path, shape_image_path, original_filename)のタプルのリスト
        - normal_image_path: 'Normal'画像の完全パス
        - shape_image_path: 'Shape'画像の完全パス
        - original_filename: 元の'Shape'画像のファイル名
    """
    normal_images = {}
    shape_images = {}
    
    # ディレクトリ内の全ファイルを走査
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "Normal" in file and file.endswith(".jpg"):
                base_name = file.replace("Normal", "")
                normal_images[base_name] = (os.path.join(root, file), file)
            elif "Shape" in file and file.endswith(".jpg"):
                base_name = file.replace("Shape", "")
                shape_images[base_name] = (os.path.join(root, file), file)
    
    # 'Normal'と'Shape'の画像ペアをマッチング
    matched_images = []
    for base_name in normal_images:
        if base_name in shape_images:
            matched_images.append((normal_images[base_name][0], 
                                 shape_images[base_name][0], 
                                 shape_images[base_name][1]))
    return matched_images

# NG画像とOK画像をそれぞれ読み込む
ng_images_label1 = load_origin_keyence_images(os.path.join(input_data_dir, "NG", ng_labels))
ok_images = load_origin_keyence_images(os.path.join(input_data_dir, "OK"))
```

```python
def visualize_image_pair(image_pairs, pair_index):
    """
    指定したペアのNormal画像とShape画像を表示します

    引数：
        image_pairs (list): (normal_path, shape_path, filename)のタプルのリスト
        pair_index (int): 表示したいペアのインデックス

    戻り値：
        None
    """
    if not image_pairs or pair_index >= len(image_pairs):
        print("指定されたインデックスの画像ペアが存在しません。")
        return
    
    # 指定されたペアの画像パスを取得
    normal_path, shape_path, filename = image_pairs[pair_index]
    
    # 画像を読み込み
    normal_img = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
    shape_img = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
    
    # 表示用のfigureを作成
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Normal画像の表示
    axes[0].imshow(normal_img, cmap='gray')
    axes[0].set_title('Normal Image')
    axes[0].axis('off')
    
    # Shape画像の表示
    axes[1].imshow(shape_img, cmap='gray')
    axes[1].set_title('Shape Image')
    axes[1].axis('off')
    
    plt.suptitle(f'Image Pair: {filename}')
    plt.tight_layout()
    plt.show()

# NG画像の最初のペアを表示
if ng_images_label1:
    print("NGデータの最初の画像ペア:")
    visualize_image_pair(ng_images_label1, 0)

# OK画像の最初のペアを表示
if ok_images:
    print("OKデータの最初の画像ペア:")
    visualize_image_pair(ok_images, 0)
```

このコードでは：

1. 画像の読み込み処理と可視化処理を分離
2. 各関数に詳細なドキュメント文字列を追加
3. エラーケースの適切な処理を追加
4. 可視化関数を再利用可能な形で実装
5. 画像表示のパラメータを適切に設定

特徴：
- 必要最小限の画像のみを表示してipynbの容量を抑制
- 画像ペアの関係性を明確に表示
- グレースケールでの表示により、輝度の違いを把握しやすく
- タイトルで画像の種類を明確に表示

コードとmarkdownを更新します。

```python
def process_images(shape_images, judge_templates, mask_templates):
    """
    全画像に対して加工領域の検出を行います
    
    引数:
        shape_images (list): Shape画像のリスト
        judge_templates (dict): 左右判定用テンプレート画像
        mask_templates (dict): マスク用テンプレート画像
        
    戻り値:
        list: (Shape画像, マスク画像, ファイル名)のタプルのリスト
    """
    processed_images = []
    for shape_path, filename in shape_images:
        shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
        # 画像の向きは揃っているので、右側のテンプレートを使用
        mask = create_processing_area_mask(shape_image, mask_templates['right'])
        processed_images.append((shape_image, mask, filename))
    
    return processed_images
```

# 4. 加工領域の特定

本セクションでは、ワークの加工領域を特定する処理について説明する。以下の3つの主要な関数で構成される：

### detect_circles()
画像から特徴的な円を検出する関数：

1. 実装内容
   - HoughCircles関数による円検出
   - 半径の大きい順に上位3つの円を選択
   - 検出円の座標と半径を取得

2. 円検出パラメータ
   - circle_dp：分解能の逆比
   - circle_min_dist：円の中心間の最小距離
   - circle_param1, circle_param2：検出感度
   - circle_min_radius, circle_max_radius：円の半径範囲

### get_optimal_scale_and_transform()
テンプレートと入力画像の位置合わせを行う関数：

1. 位置合わせ手法
   - 検出された3つの円の中心座標を使用
   - アフィン変換行列の計算
   - テンプレートから入力画像への座標変換

2. エラー処理
   - 円が3つ未満の場合はNoneを返却

### create_processing_area_mask()
加工領域のマスクを生成する関数：

1. 処理手順
   - 両画像での円検出
   - 位置合わせ用の変換行列計算
   - テンプレートの位置合わせ
   - マスク画像の生成

2. マスク生成
   - テンプレートの白い部分（輝度値128超）を検出範囲として定義
   - 二値化処理により明確な検出範囲を設定

3. エラー処理
   - 円検出失敗時は全領域をマスク対象
   - 変換行列計算失敗時も全領域をマスク対象

この実装により、画像の向きを考慮せず、単純な位置合わせとマスク生成で加工領域を特定することが可能となる。

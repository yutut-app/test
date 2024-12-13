# 2. パラメータの設定

このセクションでは、欠陥検出システムで使用する各種パラメータを定義します。適切なパラメータ設定は検出精度に大きく影響するため、以下の説明を参考に調整してください。

## 1. ディレクトリとファイル設定
```python
input_data_dir = r"../data/input"  # 入力画像のディレクトリ
output_data_dir = r"../data/output"  # 結果出力用ディレクトリ
processing_template_dir = os.path.join(input_data_dir, "mask_template")  # 加工領域テンプレート保存先
```

### テンプレートファイルパス
```python
right_processing_template_path = os.path.join(processing_template_dir, "right_processing.jpg")
left_processing_template_path = os.path.join(processing_template_dir, "left_processing.jpg")
```
- 右側と左側の加工領域のテンプレート画像のパス
- これらのテンプレートを用いて対象の加工領域を特定

### ラベル定義
```python
ng_labels = ['label1', 'label2', 'label3']  # label1: 鋳巣のみ対象
```
- 今回は鋳巣（label1）のみを検出対象とする
- label2（凹み）とlabel3（亀裂）は対象外

## 2. 円検出パラメータ
加工領域の位置合わせに使用する円検出のパラメータ
```python
circle_dp = 1.0  # 分解能の逆比
circle_min_dist = 20  # 円中心間の最小距離（px）
circle_param1 = 50  # Cannyエッジ検出の高閾値
circle_param2 = 30  # 円検出の閾値
circle_min_radius = 10  # 最小円半径（px）
circle_max_radius = 100  # 最大円半径（px）
```
- `circle_dp`: 値が大きいほど検出感度が下がり、処理が高速化
- `circle_min_dist`: 近すぎる円を別々に検出しないようにする距離
- `circle_param1`: エッジ検出感度（高いほど明確なエッジのみ検出）
- `circle_param2`: 円らしさの判定閾値（低いほど誤検出増加）
- `circle_min/max_radius`: 対象となる円のサイズ範囲

## 3. テンプレートマッチングパラメータ
```python
template_match_threshold = 0.8  # マッチング判定閾値
```
- 0-1の範囲で、高いほど厳密なマッチングを要求
- 画像の回転やスケール変化がある場合は低めに設定

## 4. スケール調整パラメータ
```python
scale_min = 0.8  # 最小スケール
scale_max = 1.2  # 最大スケール
scale_step = 0.1  # スケール調整ステップ
```
- テンプレートマッチング時の画像サイズ調整範囲
- 撮影距離のばらつきに対応

## 5. Cannyエッジ検出パラメータ（大きな欠陥用）
```python
canny_kernel_size = (3, 3)  # ガウシアンフィルタサイズ
canny_sigma = 2.0  # ガウシアンフィルタのσ
canny_min_threshold = 55  # 最小閾値
canny_max_threshold = 250  # 最大閾値
canny_merge_distance = 15  # 統合距離
texture_threshold = 30  # テクスチャ検出閾値
```
- 大きな欠陥（鋳巣）のエッジを検出するためのパラメータ
- `kernel_size`と`sigma`: ノイズ除去の強さを調整
- `min/max_threshold`: エッジの強さの判定基準
- `merge_distance`: 近接エッジの統合距離
- `texture_threshold`: テクスチャ判定の感度

## 6. DoGフィルタパラメータ（小さな欠陥用）
```python
dog_ksize = 9  # カーネルサイズ
dog_sigma1 = 1.5  # 1つ目のガウシアンσ
dog_sigma2 = 3.5  # 2つ目のガウシアンσ
dog_merge_distance = 15  # 統合距離
```
- 微細な欠陥を検出するためのDoG（Difference of Gaussian）フィルタ設定
- `sigma1`と`sigma2`の差が大きいほど、より細かい特徴を検出

## 7. 動的閾値処理パラメータ
```python
dynamic_ksize = 25  # 局所領域サイズ
dynamic_c = 6  # 閾値調整定数
dynamic_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C  # 閾値処理方法
```
- 画像の局所的な明るさの変化に適応する閾値処理
- `ksize`: 参照する周辺領域のサイズ
- `c`: 計算された閾値からのオフセット

## 8. 欠陥サイズパラメータ
```python
min_large_defect_size = 60  # 大欠陥の最小サイズ（px）
max_large_defect_size = 100  # 大欠陥の最大サイズ（px）
min_small_defect_size = 5   # 小欠陥の最小サイズ（px）
max_small_defect_size = 60  # 小欠陥の最大サイズ（px）
```
- 検出対象とする欠陥のサイズ範囲を定義
- 大小の欠陥で異なる検出処理を適用

## 9. エッジ補完パラメータ
```python
edge_kernel_size = (3, 3)  # カーネルサイズ
edge_open_iterations = 2   # オープニング回数
edge_close_iterations = 10  # クロージング回数
```
- 検出されたエッジの途切れを補完する処理のパラメータ
- オープニングでノイズ除去、クロージングで途切れを補完

## 10. マスクエッジ検出パラメータ
```python
mask_edge_min_threshold = 50  # 最小閾値
mask_edge_max_threshold = 150  # 最大閾値
mask_edge_margin = 10  # エッジ余裕幅（px）
```
- 加工領域のマスク作成時のエッジ検出パラメータ
- `margin`: エッジからの余裕幅

## 11. コントラストと輝度パラメータ
```python
min_contrast_ratio = 0.12  # 最小コントラスト比
min_intensity_diff = 25    # 最小輝度差
bright_threshold = 180     # 明領域閾値
dark_threshold = 50        # 暗領域閾値
```
- 欠陥候補領域の評価に使用する輝度関連パラメータ
- コントラスト比と輝度差で欠陥らしさを判定

## 12. ファイル出力パラメータ
```python
enlargement_factor = 1  # 拡大倍率
```
- 欠陥候補画像保存時の拡大倍率
- 視認性向上のため必要に応じて調整

## パラメータ調整のヒント
1. 検出感度を上げる場合：
   - `canny_min_threshold`を下げる
   - `dog_sigma1`と`dog_sigma2`の差を大きくする
   - `min_contrast_ratio`を下げる

2. 誤検出を減らす場合：
   - `template_match_threshold`を上げる
   - `min_defect_size`を上げる
   - `min_contrast_ratio`を上げる

3. 処理速度を上げる場合：
   - `circle_dp`を上げる
   - `scale_step`を大きくする
   - カーネルサイズを小さくする

これらのパラメータは相互に影響し合うため、検出精度と処理速度のバランスを考慮しながら調整してください。

# 7. 欠陥候補のフィルタリング

本セクションでは、検出された欠陥候補から実際の欠陥を識別し、その特徴量を計測する処理について説明する。

## パラメータ調整のポイント

1. 欠陥サイズの閾値
   - min_large_defect_size, max_large_defect_size：大きな欠陥の判定範囲
   - min_small_defect_size, max_small_defect_size：小さな欠陥の判定範囲

2. マスクエッジの処理
   - mask_edge_marginで余裕幅の大きさを調整

これらのパラメータは、実際の欠陥サイズや検査要件に応じて適切に調整する必要がある。

## 関数の説明

### extract_region_features()
検出された領域から各種特徴量を抽出する関数である：

1. 基本的な特徴量
   - 位置情報：x, y（左上座標）
   - サイズ情報：width, height, area
   - 中心位置：centroid_x, centroid_y

2. 形状特徴量
   - perimeter：周囲長
   - eccentricity：離心率
   - orientation：主軸の向き
   - major_axis_length：長軸長
   - minor_axis_length：短軸長
   - solidity：凸包に対する充填率
   - extent：バウンディングボックスに対する充填率
   - aspect_ratio：アスペクト比

3. 欠陥分類
   - max_length：最大辺長
   - detection_method：検出方法の判定
     - canny：min_large_defect_size ≤ max_length ≤ max_large_defect_size
     - dog：min_small_defect_size ≤ max_length ≤ max_small_defect_size

### create_binary_edge_image()
エッジ画像を二値化し、マスクエッジ部分を除外する関数である：

1. 処理手順
   - エッジ画像の二値化
   - マスクエッジの余裕幅の生成
   - マスクエッジ部分の除外

### filter_defects()
欠陥候補をフィルタリングし、特徴量を計測する関数である：

1. 前処理
   - エッジ画像の二値化
   - マスクエッジの除外

2. 領域解析
   - 連結成分のラベリング（8近傍接続）
   - 各領域の特徴量計測
   - サイズによる欠陥の分類（大小）

### process_completed_edges()
補完済みエッジに対してフィルタリングを実行する関数である：

1. 処理対象
   - 統合結果のエッジ
   - 大きな欠陥（Canny）のエッジ
   - 小さな欠陥（DoG）のエッジ

2. 処理内容
   - 各エッジに対する欠陥フィルタリング
   - 特徴量の計測と保存

## 可視化関数

### visualize_filtered_defects()
フィルタリング結果を可視化する関数である：

1. 表示内容
   - 元画像上に検出結果を重ねて表示
   - 大きな欠陥：赤色の矩形とラベル
   - 小さな欠陥：青色の矩形とラベル

2. 表示要素
   - バウンディングボックス
   - 欠陥ラベル（L：大きな欠陥、S：小さな欠陥）
   - 凡例


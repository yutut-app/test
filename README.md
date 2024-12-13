# 5. Canny+DoGによる欠陥候補検出

本セクションでは、鋳巣（欠陥）の候補を検出するために、大きな欠陥と小さな欠陥それぞれに適した検出手法を組み合わせて使用する。

## 検出手法の概要

1. 大きな欠陥の検出（Cannyエッジ検出）
   - サイズ: min_large_defect_size ~ max_large_defect_size
   - 特徴: エッジが明確な欠陥の検出に効果的

2. 小さな欠陥の検出（DoGフィルタ）
   - サイズ: min_small_defect_size ~ max_small_defect_size
   - 特徴: 微細な輝度変化の検出に効果的

## 関数の詳細説明

### apply_canny_detection()

大きな鋳巣を検出するためのCannyエッジ検出を実装する関数である。

1. 処理手順
   - マスク領域の抽出
   - ガウシアンブラーによるノイズ除去
   - Cannyエッジ検出
   - テクスチャ検出（Laplacian）
   - エッジの統合と近接領域の結合

2. パラメータ調整のポイント
   - canny_kernel_size: ノイズ除去の強さ
   - canny_min/max_threshold: エッジ検出の感度
   - canny_merge_distance: 近接エッジの統合範囲

### apply_dog_filter()

Difference of Gaussian（DoG）フィルタを適用する関数である。

1. 処理内容
   - 2つの異なるσ値でガウシアンブラーを適用
   - 2つのガウシアンブラー結果の差分を計算

2. パラメータ
   - dog_ksize: フィルタのサイズ
   - dog_sigma1, dog_sigma2: ガウシアンのσ値

### calculate_contrast_mask()

コントラストに基づくマスクを生成する関数である。

1. 処理内容
   - 局所的な平均値との差を計算
   - コントラスト比を算出
   - 閾値処理によるマスク生成

2. パラメータ
   - dynamic_ksize: 局所領域のサイズ
   - min_contrast_ratio: コントラスト比の閾値

### apply_dog_detection()

小さな鋳巣を検出するためのDoGフィルタ処理を実装する関数である。

1. 処理手順
   - 明暗領域の検出
   - マルチスケールDoGの適用
   - 動的閾値処理
   - コントラストマスクの生成
   - 勾配強度の計算
   - 各種マスクの統合

2. パラメータ調整のポイント
   - bright/dark_threshold: 明暗領域の閾値
   - dog_merge_distance: 近接領域の統合範囲
   - sigma_pairs: DoGのスケール設定

### calculate_gradient_magnitude()

画像の勾配強度を計算する関数である。

1. 処理内容
   - Sobelフィルタによる勾配計算
   - 勾配の大きさを計算
   - 値の正規化

### detect_defects()

CannyとDoGの両方の検出結果を統合する関数である。

1. 処理内容
   - 大きな欠陥の検出（Canny）
   - 小さな欠陥の検出（DoG）
   - 検出結果の統合

2. 戻り値
   - combined_result: 統合結果
   - large_defects: 大きな欠陥の検出結果
   - small_defects: 小さな欠陥の検出結果

## 可視化関数

### visualize_defect_detection()

検出結果を可視化する関数である。

1. 表示内容
   - 元画像
   - 大きな欠陥の検出結果（Canny）
   - 小さな欠陥の検出結果（DoG）
   - 統合結果

2. 表示形式
   - 2x2のサブプロット
   - グレースケールで表示
   - ファイル名をタイトルとして表示

## パラメータ調整の注意点

1. 欠陥サイズに応じた調整
   - min/max_large_defect_size: 大きな欠陥の検出範囲
   - min/max_small_defect_size: 小さな欠陥の検出範囲

2. 検出感度の調整
   - texture_threshold: テクスチャ検出の感度
   - dynamic_c: 動的閾値処理の感度
   - min_contrast_ratio: コントラスト検出の感度

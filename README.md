はい、各セクションのmarkdownを作成します。

# 2. パラメータの設定

1. ディレクトリとファイルパス
  - input_data_dir：入力データのルートディレクトリ
  - output_data_dir：処理結果の出力先ディレクトリ
  - template_dir：テンプレート画像の保存ディレクトリ
  - right_template_path：右側ワーク判定用のテンプレート画像パス
  - left_template_path：左側ワーク判定用のテンプレート画像パス

2. ラベル定義
  - ng_labels：欠陥種類のラベル（鋳巣、凹み、亀裂）

3. 画像処理の基本パラメータ
  - crop_width：ワーク接合部を削除する際の幅
  - threshold_value：二値化の閾値
  - kernel_size：モルフォロジー演算用のカーネルサイズ
  - iterations_open：オープニング処理の繰り返し回数
  - iterations_close：クロージング処理の繰り返し回数

4. Cannyエッジ検出のパラメータ（大きな鋳巣検出用）
  - canny_kernel_size：ガウシアンフィルタのカーネルサイズ
  - canny_sigma：ガウシアンフィルタのシグマ値
  - canny_min_threshold：エッジ検出の最小閾値
  - canny_max_threshold：エッジ検出の最大閾値
  - canny_merge_distance：検出結果の統合距離
  - texture_threshold：テクスチャ検出の閾値

5. DoGフィルタのパラメータ（小さな鋳巣検出用）
  - dog_ksize：DoGフィルタのカーネルサイズ
  - dog_sigma1：1つ目のガウシアンフィルタのシグマ値
  - dog_sigma2：2つ目のガウシアンフィルタのシグマ値
  - dog_merge_distance：検出結果の統合距離

6. 輝度ベースの検出パラメータ
  - bright_threshold：明るい領域を判定する閾値
  - dark_threshold：暗い領域を判定する閾値
  - min_contrast_ratio：最小コントラスト比

7. 動的閾値処理のパラメータ
  - dynamic_ksize：局所領域のサイズ
  - dynamic_c：閾値調整用定数
  - dynamic_method：適応的閾値処理の方法

8. エッジ補完のパラメータ
  - edge_kernel_size：エッジ補完用のカーネルサイズ
  - edge_open_iterations：ノイズ除去の繰り返し回数
  - edge_close_iterations：エッジ補完の繰り返し回数

9. マスクエッジ検出のパラメータ
  - mask_edge_min_threshold：マスクエッジ検出の最小閾値
  - mask_edge_max_threshold：マスクエッジ検出の最大閾値
  - mask_edge_margin：マスクエッジの余裕幅

10. 欠陥サイズの判定パラメータ
  - min_large_defect_size：大きな欠陥の最小サイズ
  - max_large_defect_size：大きな欠陥の最大サイズ
  - min_small_defect_size：小さな欠陥の最小サイズ
  - max_small_defect_size：小さな欠陥の最大サイズ

11. 欠陥候補の保存パラメータ
  - enlargement_factor：切り出し画像の拡大倍率

次のメッセージで#6以降のmarkdownを続けます。

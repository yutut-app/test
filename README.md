# 2. パラメータの設定

1. ディレクトリとファイルパス
  - input_data_dir：入力データのルートディレクトリ（ワークの画像が格納）
  - output_data_dir：検出結果や欠陥候補の画像、CSVファイルなどの出力先ディレクトリ
  - template_dir：ワークの向き（左右）判定用のテンプレート画像の保存ディレクトリ
  - right_template_path：右側ワーク判定用のテンプレート画像の保存パス
  - left_template_path：左側ワーク判定用のテンプレート画像の保存パス

2. ラベル定義
  - ng_labels：欠陥の種類を示すラベル
    * label1：鋳巣（内部に空洞がある欠陥）
    * label2：凹み（表面がへこんでいる欠陥）
    * label3：亀裂（表面に割れがある欠陥）

3. 画像処理の基本パラメータ
  - crop_width：画像からワーク接合部を削除する際の幅
  - threshold_value：二値化処理の閾値。この値より大きい輝度を持つピクセルを1（白）、小さい輝度を0（黒）に変換
  - kernel_size：モルフォロジー演算（膨張・収縮）で使用するカーネルの大きさ
  - iterations_open：ノイズを除去するためのオープニング処理の繰り返し回数
  - iterations_close：欠陥候補をつなげるためのクロージング処理の繰り返し回数

4. Cannyエッジ検出のパラメータ（大きな鋳巣検出用）
  - canny_kernel_size：ノイズ除去用ガウシアンフィルタのカーネルサイズ
  - canny_sigma：ガウシアンフィルタの平滑化の強さを決めるパラメータ。大きいほど強くぼかす
  - canny_min_threshold：弱いエッジとみなす最小の勾配強度
  - canny_max_threshold：強いエッジとみなす最小の勾配強度
  - canny_merge_distance：分断されたエッジをつなぐための距離
  - texture_threshold：テクスチャ（表面の凹凸）を検出する際の閾値。この値より大きい輝度変化をテクスチャとして検出

5. DoGフィルタのパラメータ（小さな鋳巣検出用）
  - dog_ksize：DoGフィルタで使用するガウシアンフィルタのカーネルサイズ
  - dog_sigma1：1つ目のガウシアンフィルタの平滑化の強さ
  - dog_sigma2：2つ目のガウシアンフィルタの平滑化の強さ。sigma1より大きい値を設定
  - dog_merge_distance：近接した検出結果をつなぐための距離

6. 輝度ベースの検出パラメータ
  - bright_threshold：明るい欠陥領域と判定する最小の輝度値
  - dark_threshold：暗い欠陥領域と判定する最大の輝度値
  - min_contrast_ratio：欠陥と判定するための最小のコントラスト比
  - min_intensity_diff：欠陥と判定するための最小の輝度差。局所的な平均輝度との差がこの値より大きい領域を欠陥候補とする

7. 動的閾値処理のパラメータ
  - dynamic_ksize：局所的な領域のサイズ。この範囲内の画素値を使って閾値を決定
  - dynamic_c：算出した閾値を調整するための定数。大きいほど検出が厳しくなる
  - dynamic_method：適応的閾値処理の方法（ガウシアン加重平均を使用）

8. エッジ補完のパラメータ
  - edge_kernel_size：エッジを補完する際に使用するカーネルサイズ）
  - edge_open_iterations：エッジのノイズを除去するためのオープニング処理の回数
  - edge_close_iterations：途切れたエッジをつなぐためのクロージング処理の回数

9. マスクエッジ検出のパラメータ
  - mask_edge_min_threshold：マスクのエッジ検出における最小閾値
  - mask_edge_max_threshold：マスクのエッジ検出における最大閾値
  - mask_edge_margin：マスクエッジから余分に確保する余白幅

10. 欠陥サイズの判定パラメータ
  - min_large_defect_size：大きな欠陥と判定する最小サイズ
  - max_large_defect_size：大きな欠陥と判定する最大サイズ
  - min_small_defect_size：小さな欠陥と判定する最小サイズ
  - max_small_defect_size：小さな欠陥と判定する最大サイズ

11. 欠陥候補の保存パラメータ
  - enlargement_factor：切り出した欠陥候補画像の拡大倍率。より詳細な確認を可能にする

継続して#6以降のmarkdownを作成しますか？

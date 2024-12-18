# ディレクトリとファイルパス
input_data_dir = r"../../data/input"
output_data_dir = r"../../data/output"
left_right_judge_template_dir = os.path.join(input_data_dir, "left_right_judge_template")
mask_template_dir = os.path.join(input_data_dir, "mask_template")

# 左右判断用テンプレートファイルパス
right_judge_template_path = os.path.join(left_right_judge_template_dir, "right_template.jpg")
left_judge_template_path = os.path.join(left_right_judge_template_dir, "left_template.jpg")

# マスク用テンプレートファイルパス
right_mask_template_path = os.path.join(mask_template_dir, "right_template.png")
left_mask_template_path = os.path.join(mask_template_dir, "left_template.png")

# ラベル定義
ng_labels = 'label1'  # label1: 鋳巣, （未実装：label2: 凹み, label3: 亀裂)
ok_labels = 'No1'  # 'No1'~'No20'

# 画像処理パラメータ
# マスク生成用パラメータ
threshold_value = 150  # 二値化の閾値（0-255）
kernel_size = (5, 5)  # モルフォロジー演算用のカーネルサイズ
iterations_open = 3  # オープニング処理の繰り返し回数（ノイズ除去用）
iterations_close = 20  # クロージング処理の繰り返し回数（穴埋め用）

# Cannyエッジ検出パラメータ（大きな鋳巣用）
canny_kernel_size = (3, 3)  # ガウシアンフィルタのカーネルサイズ
canny_sigma = 2.0  # ガウシアンフィルタのシグマ値
canny_min_threshold = 55  # Cannyの最小閾値
canny_max_threshold = 250  # Cannyの最大閾値
canny_merge_distance = 15  # Canny検出結果の統合距離
texture_threshold = 30 # テクスチャ検出用閾値

# DoGフィルタパラメータ（小さな鋳巣用）
dog_ksize = 9  # DoGフィルタのカーネルサイズ
dog_sigma1 = 1.5  # 1つ目のガウシアンフィルタのシグマ値
dog_sigma2 = 3.5  # 2つ目のガウシアンフィルタのシグマ値
dog_merge_distance = 15  # DoG検出結果の統合距離

# 輝度ベースの検出パラメータ
bright_threshold = 180  # 明るい欠陥領域の閾値
dark_threshold = 50  # 暗い欠陥領域の閾値
min_intensity_diff = 25  # 最小輝度差
min_contrast_ratio = 0.12  # 最小コントラスト比

# 動的閾値処理パラメータ
dynamic_ksize = 25  # 局所領域のサイズ
dynamic_c = 6  # 閾値調整用定数
dynamic_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C  # 適応的閾値処理の方法

# エッジ補完パラメータ
edge_kernel_size = (3, 3)  # エッジ補完のカーネルサイズ
edge_open_iterations = 2   # ノイズ削除の繰り返し回数
edge_close_iterations = 10  # エッジ補完の繰り返し回数

# マスクエッジ検出パラメータ
mask_edge_min_threshold = 50  # マスクエッジ検出の最小閾値
mask_edge_max_threshold = 150  # マスクエッジ検出の最大閾値
mask_edge_margin = 10  # マスクエッジの余裕幅（ピクセル）

# 欠陥サイズパラメータ
min_large_defect_size = 60  # 大きな欠陥の最小サイズ（ピクセル）
max_large_defect_size = 100  # 大きな欠陥の最大サイズ（ピクセル）
min_small_defect_size = 5   # 小さな欠陥の最小サイズ（ピクセル）
max_small_defect_size = 60  # 小さな欠陥の最大サイズ（ピクセル）

# 欠陥候補の保存パラメータ
enlargement_factor = 1  # 切り出し画像の拡大倍率(基本は1から変えない、拡大して保存したい場合に変更)

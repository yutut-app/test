# ディレクトリとファイルパス
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"
left_right_judge_template_dir = os.path.join(input_data_dir, "left_right_judge_template")
mask_template_dir = os.path.join(input_data_dir, "mask_template")

# 左右判断用テンプレートファイルパス
right_judge_template_path = os.path.join(left_right_judge_template_dir, "right_template.jpg")
left_judge_template_path = os.path.join(left_right_judge_template_dir, "left_template.jpg")

# マスク用テンプレートファイルパス
right_mask_template_path = os.path.join(mask_template_dir, "right_template.jpg")
left_mask_template_path = os.path.join(mask_template_dir, "left_template.jpg")

# ラベル定義
ng_labels = 'label1'  # label1: 鋳巣, （未実装：label2: 凹み, label3: 亀裂)
ok_labels = 'No1'  # 'No1'~'No20'

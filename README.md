import os

# 欠陥候補を外接矩形で切り出し、保存する関数
def save_defect_candidates(image, defects, image_name, output_dir):
    # 欠陥候補用ディレクトリの作成
    defect_dir = os.path.join(output_dir, "defect_candidate")
    if not os.path.exists(defect_dir):
        os.makedirs(defect_dir)
    
    # 元の画像ごとにディレクトリ作成
    image_dir = os.path.join(defect_dir, image_name)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # 欠陥候補を保存
    for i, (x, y, w, h, cx, cy) in enumerate(defects):
        # 外接矩形で欠陥候補を切り出し
        defect_candidate = image[y:y+h, x:x+w]
        
        # 欠陥候補の拡大（100倍）
        enlarged_defect = cv2.resize(defect_candidate, (w * 100, h * 100), interpolation=cv2.INTER_NEAREST)
        
        # 保存ファイルパス
        candidate_filename = os.path.join(image_dir, f"defect_{i+1}.png")
        
        # 保存処理
        cv2.imwrite(candidate_filename, enlarged_defect)

# 欠陥候補を保存する処理を統合
def process_and_save_defects(labeled_images, original_images, output_dir):
    for (binarized_image, edge_image, defects), original_image_path in zip(labeled_images, original_images):
        # 元画像のファイル名を取得
        image_name = os.path.splitext(os.path.basename(original_image_path))[0]
        
        # 元画像を読み込む
        original_image = cv2.imread(original_image_path)
        
        # 欠陥候補を保存
        save_defect_candidates(original_image, defects, image_name, output_dir)

# NGとOK画像に対して保存処理を実行
process_and_save_defects(labeled_ng_images_label1, [img[0] for img in ng_images_label1], output_data_dir)
process_and_save_defects(labeled_ng_images_label2, [img[0] for img in ng_images_label2], output_data_dir)
process_and_save_defects(labeled_ng_images_label3, [img[0] for img in ng_images_label3], output_data_dir)
process_and_save_defects(labeled_ok_images, [img[0] for img in ok_images], output_data_dir)

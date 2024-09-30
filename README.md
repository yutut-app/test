import os

# 欠陥候補を保存する関数
def save_defect_candidates(image, defects, image_name, save_dir):
    # 保存先のディレクトリがなければ作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 欠陥候補ごとに外接矩形で画像を切り出して保存
    for idx, (x, y, w, h, cx, cy) in enumerate(defects):
        # 外接矩形で画像を切り出し
        defect_image = image[y:y+h, x:x+w]
        
        # 保存ファイル名
        save_path = os.path.join(save_dir, f"{image_name}_defect_{idx}.jpg")
        
        # 画像を保存
        cv2.imwrite(save_path, defect_image)

# 全ての欠陥候補を保存する処理を追加
def save_defects_for_all_images(labeled_images, original_images, output_dir):
    save_dir = os.path.join(output_dir, "defect_candidate")
    
    # 画像ごとに欠陥候補を保存
    for idx, (binarized_image, edge_image, defects) in enumerate(labeled_images):
        if defects:
            # 元画像名を取得して保存名に使用
            original_image_path = original_images[idx][0]  # 元画像
            image_name = os.path.splitext(os.path.basename(original_image_path))[0]
            
            # 欠陥候補を保存
            save_defect_candidates(edge_image, defects, image_name, save_dir)

# NGとOK画像に対して欠陥候補を保存
save_defects_for_all_images(labeled_ng_images_label1, ng_images_label1, output_data_dir)
save_defects_for_all_images(labeled_ng_images_label2, ng_images_label2, output_data_dir)
save_defects_for_all_images(labeled_ng_images_label3, ng_images_label3, output_data_dir)
save_defects_for_all_images(labeled_ok_images, ok_images, output_data_dir)

def save_defect_image(image, defect, output_dir, image_name, defect_number):
    x1, y1 = defect['x'], defect['y']
    x2 = x1 + defect['width']
    y2 = y1 + defect['height']
    
    # 欠陥候補部分の四角形で画像を切り出し
    defect_image = image[y1:y2, x1:x2]
    enlarged_image = cv2.resize(defect_image, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    
    output_filename = f"defect_{defect_number}.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, enlarged_image)
    
    return output_filename

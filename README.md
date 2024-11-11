# 6. DoGフィルタと動的閾値処理による欠陥検出

def difference_of_gaussian(img, ksize, sigma1, sigma2):
    # ガウシアンフィルタ適用
    gaussian_1 = cv2.GaussianBlur(img, (ksize, ksize), sigma1)
    gaussian_2 = cv2.GaussianBlur(img, (ksize, ksize), sigma2)
    
    # 2種のガウシアンフィルタ適用画像の差分
    dog = gaussian_1 - gaussian_2
    
    return dog

def dynamic_threshold(img, ksize, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, c=2):
    # 適応的閾値処理
    binary = cv2.adaptiveThreshold(img, 255, method, cv2.THRESH_BINARY_INV, ksize, c)
    return binary

def detect_defects_dog_dynamic(cropped_keyence_image, binarized_image):
    # DoGフィルタ適用
    dog_result = difference_of_gaussian(cropped_keyence_image, dog_ksize, dog_sigma1, dog_sigma2)
    
    # DoG結果を8ビット unsigned int に変換
    dog_result = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 動的閾値処理
    dynamic_result = dynamic_threshold(cropped_keyence_image, dynamic_ksize, dynamic_method, dynamic_c)
    
    # DoGと動的閾値の結果を組み合わせる
    combined_result = cv2.bitwise_and(dog_result, dynamic_result)
    
    # マスク適用
    masked_result = cv2.bitwise_and(combined_result, combined_result, mask=binarized_image)
    
    return masked_result

def process_images_for_defect_detection(binarized_images):
    processed_images = []
    for binarized_image, cropped_keyence_image, original_filename in binarized_images:
        defect_image = detect_defects_dog_dynamic(cropped_keyence_image, binarized_image)
        processed_images.append((binarized_image, defect_image, original_filename))
    return processed_images

# NGとOK画像に対して欠陥検出を実行
processed_ng_images_label1 = process_images_for_defect_detection(binarized_ng_images_label1)
processed_ng_images_label2 = process_images_for_defect_detection(binarized_ng_images_label2)
processed_ng_images_label3 = process_images_for_defect_detection(binarized_ng_images_label3)
processed_ok_images = process_images_for_defect_detection(binarized_ok_images)

def save_defect_image(image, defect, output_dir, image_name, defect_number):
    cx, cy = defect['centroid_x'], defect['centroid_y']
    size = max(defect['width'], defect['height'])
    
    x1 = max(int(cx - size), 0)
    y1 = max(int(cy - size), 0)
    x2 = min(int(cx + size), image.shape[1])
    y2 = min(int(cy + size), image.shape[0])
    
    defect_image = image[y1:y2, x1:x2]
    enlarged_image = cv2.resize(defect_image, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
    
    output_filename = f"defect_{defect_number}.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, enlarged_image)
    
    return output_filename

def process_images_for_saving(filtered_images, base_output_dir, image_label, shape_images):
    all_defects_data = []
    
    for image_name, binarized_image, edge_image, defects in filtered_images:
        image_type = image_name.split('_')[0]
        output_dir = os.path.join(base_output_dir, image_type, image_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # shape_imagesからも欠陥を切り出す
        original_image_path = shape_images[image_name]
        original_image = cv2.imread(original_image_path)
        shape_output_dir = os.path.join(output_dir, 'shape_images')
        os.makedirs(shape_output_dir, exist_ok=True)
        
        for defect in defects:
            output_filename = save_defect_image(edge_image, defect, output_dir, image_name, defect['label'])
            shape_output_filename = save_defect_image(original_image, defect, shape_output_dir, image_name, defect['label'])
            
            defect_data = {
                'image_name': image_name,
                'defect_image': os.path.join(image_type, image_name, output_filename),
                'shape_defect_image': os.path.join(image_type, image_name, 'shape_images', shape_output_filename),
                'Image_label': image_label,
                'defect_label': 0,  # デフォルトで0（OK）とする
            }
            defect_data.update(defect)
            all_defects_data.append(defect_data)
    
    return all_defects_data

# NGとOK画像の欠陥候補を保存し、データを収集
output_dir = os.path.join(output_data_dir, "defect_images")
os.makedirs(output_dir, exist_ok=True)

all_defects_data = []
all_defects_data.extend(process_images_for_saving(filtered_ng_images_label1, output_dir, 1, shape_images))
all_defects_data.extend(process_images_for_saving(filtered_ng_images_label2, output_dir, 1, shape_images))
all_defects_data.extend(process_images_for_saving(filtered_ng_images_label3, output_dir, 1, shape_images))
all_defects_data.extend(process_images_for_saving(filtered_ok_images, output_dir, 0, shape_images))

# CSVファイルに出力
df = pd.DataFrame(all_defects_data)
csv_output_path = os.path.join(output_data_dir, "defects_data.csv")
df.to_csv(csv_output_path, index=False)

print(f"Defects data saved to {csv_output_path}")


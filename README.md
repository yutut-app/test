import gc

def process_images(image_pairs, batch_size=100):
    updated_images = []
    num_images = len(image_pairs)
    
    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)
        batch_pairs = image_pairs[start:end]
        
        for origin_image_path, keyence_image_path in batch_pairs:
            cropped_image, cropped_keyence_image = remove_joint_part(origin_image_path, keyence_image_path)
            updated_images.append((cropped_image, cropped_keyence_image))
        
        # メモリを解放
        del batch_pairs
        gc.collect()  # Pythonのガベージコレクションを手動で呼び出してメモリを解放
        
        print(f"Processed {end}/{num_images} images")
    
    return updated_images

# NGとOK画像に対してバッチ処理を実行
updated_ok_images = process_images(ok_images, batch_size=100)

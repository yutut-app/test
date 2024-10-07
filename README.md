import os
import cv2
import numpy as np
from skimage import io
import gc

def load_image(image_path):
    try:
        return io.imread(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def process_single_image(image_pair):
    origin_image_path, keyence_image_path = image_pair
    origin_image = load_image(origin_image_path)
    keyence_image = load_image(keyence_image_path)
    
    if origin_image is None or keyence_image is None:
        return None
    
    cropped_image, cropped_keyence_image = remove_joint_part(origin_image, keyence_image)
    
    # メモリ解放
    del origin_image, keyence_image
    gc.collect()
    
    return cropped_image, cropped_keyence_image

def process_images_in_batches(image_paths, process_func, batch_size=BATCH_SIZE):
    all_results = []
    total = len(image_paths)
    
    for i in range(0, total, batch_size):
        batch = image_paths[i:i+batch_size]
        batch_results = [process_func(pair) for pair in batch if pair is not None]
        batch_results = [result for result in batch_results if result is not None]
        all_results.extend(batch_results)
        
        # メモリ解放
        del batch_results
        gc.collect()
        
        # 進捗状況の表示
        print(f"Processed {min(i+batch_size, total)}/{total} images")
    
    return all_results

def load_and_process_images(directory):
    image_pairs = load_origin_keyence_images(directory)
    return process_images_in_batches(image_pairs, process_single_image)

# メイン処理
ng_images_label1 = load_and_process_images(os.path.join(input_data_dir, "NG", "label1"))
ng_images_label2 = load_and_process_images(os.path.join(input_data_dir, "NG", "label2"))
ng_images_label3 = load_and_process_images(os.path.join(input_data_dir, "NG", "label3"))
ok_images = load_and_process_images(os.path.join(input_data_dir, "OK"))

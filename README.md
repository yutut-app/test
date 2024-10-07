def process_images_in_batches(image_pairs, batch_size=10):
    updated_images = []
    for i in range(0, len(image_pairs), batch_size):
        batch = image_pairs[i:i+batch_size]
        batch_results = process_images(batch)
        updated_images.extend(batch_results)
        print(f"Processed batch {i//batch_size + 1}/{-(-len(image_pairs)//batch_size)}")
    return updated_images

def process_images(image_pairs):
    updated_images = []
    for origin_image_path, keyence_image_path in image_pairs:
        try:
            cropped_image, cropped_keyence_image = remove_joint_part(origin_image_path, keyence_image_path)
            updated_images.append((cropped_image, cropped_keyence_image))
        except Exception as e:
            print(f"Error processing images {origin_image_path} and {keyence_image_path}: {str(e)}")
            continue
    return updated_images

# メイン処理部分の変更
batch_size = 50  # バッチサイズを設定（必要に応じて調整してください）

# NGとOK画像に対して接合部削除を実行
updated_ng_images_label1 = process_images_in_batches(ng_images_label1, batch_size)
updated_ng_images_label2 = process_images_in_batches(ng_images_label2, batch_size)
updated_ng_images_label3 = process_images_in_batches(ng_images_label3, batch_size)
updated_ok_images = process_images_in_batches(ok_images, batch_size)

# メモリ解放
del ng_images_label1, ng_images_label2, ng_images_label3, ok_images

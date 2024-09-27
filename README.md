# Display first NG images from label1 (porosity) - both origin and Keyence processed images
if ng_images_label1:
    # Get the first pair of Normal and Shape images
    origin_image_path, keyence_image_path = ng_images_label1[0]
    
    # Load the images
    origin_image = io.imread(origin_image_path)
    keyence_image = io.imread(keyence_image_path)

    # Display original image (Normal)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(origin_image, cmap='gray')
    plt.title(f"Origin Image - {os.path.basename(origin_image_path)}")
    plt.axis('off')

    # Display Keyence processed image (Shape)
    plt.subplot(1, 2, 2)
    plt.imshow(keyence_image, cmap='gray')
    plt.title(f"Keyence Image - {os.path.basename(keyence_image_path)}")
    plt.axis('off')

    plt.show()
else:
    print("No images found in NG_label1.")



# 更新されたNG_label1の最初の画像ペアを表示
if updated_ng_images_label1:
    cropped_origin_image, cropped_keyence_image = updated_ng_images_label1[0]
    
    # 切り取った元画像の表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cropped_origin_image, cmap='gray')
    plt.title("Cropped Origin Image")
    plt.axis('off')

    # 切り取ったキーエンス画像の表示
    plt.subplot(1, 2, 2)
    plt.imshow(cropped_keyence_image, cmap='gray')
    plt.title("Cropped Keyence Image")
    plt.axis('off')

    plt.show()
else:
    print("No images found after processing.")






# 更新されたNG_label1の最初の画像ペアを表示
if binarized_ng_images_label1:
    binarized_image, cropped_keyence_image = binarized_ng_images_label1[0]
    
    # 二直化後の画像の表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(binarized_image, cmap='gray')
    plt.title("Binarized Origin Image")
    plt.axis('off')

    # キーエンス画像の表示
    plt.subplot(1, 2, 2)
    plt.imshow(cropped_keyence_image, cmap='gray')
    plt.title("Cropped Keyence Image")
    plt.axis('off')

    plt.show()
else:
    print("No images found after binarization.")







# 更新されたNG_label1の最初の画像ペアを表示
if edged_ng_images_label1:
    binarized_image, edge_image = edged_ng_images_label1[0]
    
    # 二直化後の画像の表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(binarized_image, cmap='gray')
    plt.title("Binarized Image")
    plt.axis('off')

    # エッジ検出後の画像の表示
    plt.subplot(1, 2, 2)
    plt.imshow(edge_image, cmap='gray')
    plt.title("Edge Detection Image")
    plt.axis('off')

    plt.show()
else:
    print("No images found after edge detection.")

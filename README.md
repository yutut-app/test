def filter_defects_by_max_length(defects, min_size, max_size):
    return [defect for defect in defects if min_size <= defect['max_length'] <= max_size]

def process_images_for_filtering(labeled_images, image_type):
    filtered_images = []
    for i, (binarized_image, edge_image, defects) in enumerate(labeled_images):
        # 大きな欠陥（Cannyで検出）のフィルタリング
        large_defects = filter_defects_by_max_length(defects, min_large_defect_size, max_large_defect_size)
        for j, defect in enumerate(large_defects, 1):
            defect['label'] = j
            defect['detection_type'] = 'canny'  # Cannyで検出したことを記録
        
        # 小さな欠陥（DoGで検出）のフィルタリング
        small_defects = filter_defects_by_max_length(defects, min_small_defect_size, max_small_defect_size)
        for j, defect in enumerate(small_defects, len(large_defects) + 1):
            defect['label'] = j
            defect['detection_type'] = 'dog'  # DoGで検出したことを記録
        
        # 大きな欠陥と小さな欠陥を統合
        filtered_defects = large_defects + small_defects
        
        image_name = f"{image_type}_{i}"
        filtered_images.append((image_name, binarized_image, edge_image, filtered_defects))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images_label1 = process_images_for_filtering(labeled_ng_images_label1, "ng_label1")
filtered_ng_images_label2 = process_images_for_filtering(labeled_ng_images_label2, "ng_label2")
filtered_ng_images_label3 = process_images_for_filtering(labeled_ng_images_label3, "ng_label3")
filtered_ok_images = process_images_for_filtering(labeled_ok_images, "ok")

# フィルタリング結果の可視化
def visualize_filtered_defects(image_name, image, defects, mask):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image, cmap='gray')
    
    # マスクのエッジを可視化
    mask_edges = create_mask_edge_margin(mask, mask_edge_margin)
    ax.imshow(mask_edges, alpha=0.3, cmap='cool')
    
    # 検出方法によって色を変える
    colors = {
        'canny': 'red',    # Cannyで検出した欠陥は赤色
        'dog': 'blue'      # DoGで検出した欠陥は青色
    }
    
    for defect in defects:
        color = colors.get(defect['detection_type'], 'yellow')  # デフォルトは黄色
        rect = plt.Rectangle((defect['x'], defect['y']), defect['width'], defect['height'],
                           fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(defect['x'], defect['y'], str(defect['label']), 
                color=color, fontsize=12)
    
    # 凡例の追加
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='red', label='Canny (Large Defects)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='blue', label='DoG (Small Defects)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.title(f"Filtered Defects with Mask Edges - {image_name}", fontsize=20)
    plt.axis('off')
    plt.show()

# フィルタリング結果の可視化（例：最初のNG画像）
if filtered_ng_images_label1:
    image_name, binarized_image, edge_image, filtered_defects = filtered_ng_images_label1[0]
    visualize_filtered_defects(image_name, edge_image, filtered_defects, binarized_image)

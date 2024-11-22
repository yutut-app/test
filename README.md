Cannyで検出した大きな鋳巣を赤色、DoGで検出した小さな鋳巣を青色で表示するように改良します。フィルタリング結果の可視化部分のコードを変更するため、#7、#8、#9の全体を記載します。

該当の変更があるのは#8のvisualize_filtered_defectsですが、関連する処理の変更も必要なため、全体を記載します。

```python
# 7. エッジの補完とラベリング処理

def create_mask_edge_margin(mask, margin):
    mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
    dilated_edges = cv2.dilate(mask_edges, kernel, iterations=1)
    return dilated_edges

def complete_edges(edge_image, mask):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    skeleton = skeletonize(edge_image > 0)
    kernel = np.ones(edge_kernel_size, np.uint8)
    
    # オープン処理でノイズを削除
    opened_skeleton = cv2.morphologyEx(skeleton.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=edge_open_iterations)
    
    # クローズ処理でエッジを接続
    connected_skeleton = cv2.morphologyEx(opened_skeleton, cv2.MORPH_CLOSE, kernel, iterations=edge_close_iterations)
    
    completed_edges = np.maximum(edge_image, connected_skeleton * 255)
    completed_edges = np.where(mask_edges_with_margin > 0, edge_image, completed_edges)
    return completed_edges.astype(np.uint8)

def process_images_for_edge_completion(processed_images):
    completed_edge_images = []
    for binarized_image, edge_image, original_filename in processed_images:
        completed_edges = complete_edges(edge_image, binarized_image)
        completed_edge_images.append((binarized_image, completed_edges, original_filename))
    return completed_edge_images

# NGとOK画像に対してエッジ補完を実行
completed_ng_images_label1 = process_images_for_edge_completion(processed_ng_images_label1)
completed_ng_images_label2 = process_images_for_edge_completion(processed_ng_images_label2)
completed_ng_images_label3 = process_images_for_edge_completion(processed_ng_images_label3)
completed_ok_images = process_images_for_edge_completion(processed_ok_images)

# 8. 欠陥候補のフィルタリング

def filter_and_measure_defects(edge_image, mask, min_size, max_size):
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    binary_edge_image = (edge_image > 0).astype(np.uint8)
    binary_edge_image[mask_edges_with_margin > 0] = 0  # マスクエッジ部分を除外
    
    labels = measure.label(binary_edge_image, connectivity=2)
    defects = []
    
    for region in measure.regionprops(labels):
        if min_size <= region.area <= max_size:
            y, x = region.bbox[0], region.bbox[1]
            h, w = region.bbox[2] - y, region.bbox[3] - x
            defect_info = {
                'label': region.label,
                'x': x, 'y': y, 'width': w, 'height': h,
                'area': region.area,
                'centroid_y': region.centroid[0], 'centroid_x': region.centroid[1],
                'perimeter': region.perimeter,
                'eccentricity': region.eccentricity,
                'orientation': region.orientation,
                'major_axis_length': region.major_axis_length,
                'minor_axis_length': region.minor_axis_length,
                'solidity': region.solidity,
                'extent': region.extent,
                'aspect_ratio': max(w, h) / min(w, h) if min(w, h) > 0 else 0,
                'max_length': max(w, h),
                'detection_method': 'canny' if max(w, h) >= min_large_defect_size else 'dog'  # 検出方法を追加
            }
            defects.append(defect_info)
    
    return defects

def process_images_for_filtering(completed_edge_images):
    filtered_images = []
    for binarized_image, edge_image, original_filename in completed_edge_images:
        defects = filter_and_measure_defects(edge_image, binarized_image, min_small_defect_size, max_large_defect_size)
        filtered_images.append((original_filename, binarized_image, edge_image, defects))
    return filtered_images

# NGとOK画像に対してフィルタリングを実行
filtered_ng_images_label1 = process_images_for_filtering(completed_ng_images_label1)
filtered_ng_images_label2 = process_images_for_filtering(completed_ng_images_label2)
filtered_ng_images_label3 = process_images_for_filtering(completed_ng_images_label3)
filtered_ok_images = process_images_for_filtering(completed_ok_images)

def visualize_filtered_defects(image_name, image, defects, mask):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image, cmap='gray')
    
    # マスクのエッジを可視化
    mask_edges = create_mask_edge_margin(mask, mask_edge_margin)
    ax.imshow(mask_edges, alpha=0.3, cmap='cool')
    
    # 凡例用のパッチを作成
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor='red', label='Large defect (Canny)'),
        patches.Patch(facecolor='none', edgecolor='blue', label='Small defect (DoG)')
    ]
    
    for defect in defects:
        # 検出方法に応じて色を選択
        color = 'red' if defect['detection_method'] == 'canny' else 'blue'
        
        rect = plt.Rectangle((defect['x'], defect['y']), defect['width'], defect['height'],
                           fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(defect['x'], defect['y'], str(defect['label']), color=color, fontsize=12)
    
    # 凡例を追加
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.title(f"Filtered Defects with Mask Edges - {image_name}\nRed: Canny (Large), Blue: DoG (Small)", fontsize=20)
    plt.axis('off')
    plt.show()

# フィルタリング結果の可視化（例：最初のNG画像）
if filtered_ng_images_label1:
    image_name, binarized_image, edge_image, filtered_defects = filtered_ng_images_label1[0]
    visualize_filtered_defects(image_name, edge_image, filtered_defects, binarized_image)
```

```python
# 9. 欠陥候補の画像の保存とCSV出力

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

def process_images_for_saving(filtered_images, base_output_dir, image_label):
    if filtered_images is None:
        return []  # フィルタリングされた画像がない場合は空のリストを返す
    
    defects_data = []
    
    for original_filename, binarized_image, edge_image, defects in filtered_images:
        image_type = "ng" if image_label == 1 else "ok"
        output_dir = os.path.join(base_output_dir, image_type, original_filename.replace(".jpg", ""))
        os.makedirs(output_dir, exist_ok=True)
        
        for defect in defects:
            output_filename = save_defect_image(edge_image, defect, output_dir, original_filename, defect['label'])
            
            defect_data = {
                'image_name': original_filename,
                'defect_image': os.path.join(image_type, original_filename.replace(".jpg", ""), output_filename),
                'Image_label': image_label,
                'defect_label': 0,  # デフォルトで0（OK）とする
                'detection_method': defect['detection_method']  # 検出方法を追加
            }
            defect_data.update(defect)
            defects_data.append(defect_data)
    
    return defects_data

# メインの処理部分
output_dir = os.path.join(output_data_dir, "defect_images")
os.makedirs(output_dir, exist_ok=True)

all_defects_data = []
for name, filtered_images, label in [
    ("NG Label 1", filtered_ng_images_label1, 1),
    ("NG Label 2", filtered_ng_images_label2, 1),
    ("NG Label 3", filtered_ng_images_label3, 1),
    ("OK", filtered_ok_images, 0)
]:
    if filtered_images:
        print(f"Processing {name} images...")
        all_defects_data.extend(process_images_for_saving(filtered_images, output_dir, label))
    else:
        print(f"No data for {name} images.")

# CSVファイルに出力
csv_output_dir = os.path.join(output_data_dir, "defect_data")
os.makedirs(csv_output_dir, exist_ok=True)
csv_output_path = os.path.join(csv_output_dir, "defects_data.csv")

df = pd.DataFrame(all_defects_data)

if os.path.exists(csv_output_path):
    # 既存のCSVファイルが存在する場合、列名なしで上書き
    df.to_csv(csv_output_path, mode='a', header=False, index=False)
    print(f"Appended defects data to existing file: {csv_output_path}")
else:
    # 新規作成の場合、列名ありで保存
    df.to_csv(csv_output_path, index=False)
    print(f"Created new defects data file: {csv_output_path}")

print(f"Defects data saved to {csv_output_path}")
```

主な変更点：
1. defect_infoに'detection_method'を追加し、Cannyで検出したか、DoGで検出したかを記録
2. visualize_filtered_defectsを改良し、検出方法に応じて色分け表示（Canny:赤、DoG:青）
3. 凡例を追加して検出方法を明示
4. CSVファイルにdetection_methodカラムを追加

これにより、可視化結果で各欠陥がどちらの手法で検出されたかが一目でわかるようになります。

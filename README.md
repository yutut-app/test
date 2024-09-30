```python
# パラメータの設定
input_data_dir = r"../data/input"
output_data_dir = r"../data/output"
ng_labels = ['label1', 'label2', 'label3']  # label1: 鋳巣, label2: 凹み, label3: 亀裂
template_dir = os.path.join(input_data_dir, "template")
right_template_path = os.path.join(template_dir, "right_keyence.jpg")
left_template_path = os.path.join(template_dir, "left_keyence.jpg")
crop_width = 1360  # ワーク接合部を削除するための幅
threshold_value = 150  # 二直化しきい値
kernel_size = (5, 5)  # カーネルサイズ
iterations_open = 3  # 膨張処理の繰り返し回数
iterations_close = 20  # 収縮処理の繰り返し回数
gaussian_kernel_size = (7, 7)  # ガウシアンブラーのカーネルサイズ
canny_min_threshold = 30  # エッジ検出の最小しきい値
canny_max_threshold = 120  # エッジ検出の最大しきい値
sigma = 3  # ガウシアンブラーの標準偏差
min_defect_size = 5  # 最小欠陥サイズ（0.5mm = 5px）
max_defect_size = 100  # 最大欠陥サイズ（10mm = 100px）
texture_threshold = 15  # テクスチャの変化を検出するためのしきい値
edge_margin = 5  # マスクのエッジ部分に持たせる余裕（ピクセル単位）
edge_completion_kernel_size = (3, 3)  # エッジ補完に使用するカーネルサイズ
canny_edge_min = 100  # フィルタリング時のCannyエッジ検出の最小しきい値
canny_edge_max = 200  # フィルタリング時のCannyエッジ検出の最大しきい値

```

---

## 7. エッジの補完と欠陥候補の中心座標の取得
### 目的:
欠陥候補のエッジを補完し、隣接するピクセルのグループを一つの領域として識別します。その後、欠陥候補の中心座標と特徴量を抽出し、ラベリングされた画像を可視化します。

```python
def complete_edges_and_extract_defects(edge_image, binarized_image, kernel_size):
    # 膨張・収縮処理でエッジを補完
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    completed_edges = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # ラベリング処理
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(completed_edges)
    
    defect_candidates = []
    for i in range(1, num_labels):  # ラベル0は背景
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        defect_candidates.append((x, y, w, h, cx, cy))
    
    # ラベリングされた画像を可視化
    labeled_image = cv2.cvtColor(binarized_image, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h, cx, cy) in defect_candidates:
        cv2.rectangle(labeled_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(labeled_image, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    
    return labeled_image, defect_candidates

```

---

## 8. 欠陥候補のフィルタリング
### 目的:
マスクのエッジと重なっている欠陥候補を除外し、各ラベル領域のサイズをチェックして指定範囲内の欠陥候補のみを抽出します。さらに、欠陥候補の中心を基に200px x 200pxのバウンディングボックスを画像内に加え、ラベル付けされた画像を可視化します。

```python
def filter_defects(defects, mask, min_size, max_size, canny_edge_min, canny_edge_max):
    filtered_defects = []
    mask_edges = cv2.Canny(mask, canny_edge_min, canny_edge_max)
    
    # マスクエッジの可視化
    plt.figure(figsize=(10, 5))
    plt.imshow(mask_edges, cmap='gray')
    plt.title("Mask Edges")
    plt.axis('off')
    plt.show()
    
    for (x, y, w, h, cx, cy) in defects:
        # 欠陥候補の外接矩形がエッジと重なっているか確認
        if np.any(mask_edges[y:y+h, x:x+w] > 0):
            continue
        
        # サイズフィルタリング
        if min_size <= w <= max_size and min_size <= h <= max_size:
            filtered_defects.append((x, y, w, h, cx, cy))
    
    return filtered_defects

def draw_bounding_boxes(image, defects):
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h, cx, cy) in defects:
        # 中心座標を基に200px x 200pxの正方形バウンディングボックスを描画
        half_size = 100  # 200px x 200pxの半分
        top_left = (int(cx) - half_size, int(cy) - half_size)
        bottom_right = (int(cx) + half_size, int(cy) + half_size)
        cv2.rectangle(result_image, top_left, bottom_right, (255, 0, 0), 2)
    
    return result_image
```

---

## 9. 欠陥候補の画像の保存
### 目的:
欠陥候補の中心座標を基に200px x 200pxの正方形として切り出し、それを10倍に拡大して保存します。さらに、欠陥候補の情報をCSVに保存します。

```python
# 欠陥候補を保存する関数 (cropped_keyence_image からの切り出し)
def save_defect_images_and_csv(defects, cropped_keyence_image, image_name, output_dir, image_label):
    defect_dir = os.path.join(output_dir, "defect_candidate", image_name)
    if not os.path.exists(defect_dir):
        os.makedirs(defect_dir)
    
    defect_data = []
    
    for i, (x, y, w, h, cx, cy) in enumerate(defects):
        # 欠陥候補の200px x 200pxの正方形切り出し
        half_size = 100
        top_left_x = int(cx) - half_size
        top_left_y = int(cy) - half_size
        bottom_right_x = int(cx) + half_size
        bottom_right_y = int(cy) + half_size
        
        # キャンバスの作成: RGB画像として黒色のキャンバス (エラー解消のため)
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)  # 黒色の200px x 200pxキャンバス
        
        # 切り出す範囲を調整
        cut_top_left_x = max(0, top_left_x)
        cut_top_left_y = max(0, top_left_y)
        cut_bottom_right_x = min(cropped_keyence_image.shape[1], bottom_right_x)
        cut_bottom_right_y = min(cropped_keyence_image.shape[0], bottom_right_y)
        
        # cropped_keyence_imageから切り出し
        cropped_part = cropped_keyence_image[cut_top_left_y:cut_bottom_right_y, cut_top_left_x:cut_bottom_right_x]
        
        # キャンバスの対応する位置に欠陥部分を貼り付け (グレースケール画像にRGB画像を保持するための対応)
        start_x = max(0, -top_left_x)
        start_y = max(0, -top_left_y)
        
        # エラー修正: モノクロ画像にも対応
        if cropped_part.ndim == 2:
            cropped_part = cv2.cvtColor(cropped_part, cv2.COLOR_GRAY2RGB)
        
        # キャンバスの該当部分に欠陥部分を埋め込む
        canvas[start_y:start_y + cropped_part.shape[0], start_x:start_x + cropped_part.shape[1]] = cropped_part
        
        # 10倍に拡大
        enlarged_image = cv2.resize(canvas, (2000, 2000), interpolation=cv2.INTER_LINEAR)
        
        # 画像の保存
        defect_filename = f"defect_{i+1}.png"
        defect_filepath = os.path.join(defect_dir, defect_filename)
        cv2.imwrite(defect_filepath, enlarged_image)
        
        # CSV用データの収集
        defect_data.append({
            "image_name": image_name,
            "width": float(w),
            "height": float(h),
            "center_x": float(cx),
            "center_y": float(cy),
            "Image_label": image_label,
            "defect_label": 0  # 欠陥ラベルは全て0 (OK)として保存
        })
    
    # CSVの保存
    csv_filepath = os.path.join(defect_dir, f"{image_name}_defects.csv")
    df = pd.DataFrame(defect_data)
    df.to_csv(csv_filepath, index=False)

```

---

## 全体の処理の統合
NGとOK画像に対してエッジ検出から欠陥候補の保存までの処理を統合します。

```python
def process_and_save_defects(labeled_images, original_images, output_dir, image_label):
    for (binarized_image, edge_image, defects), original_image_path in zip(labeled_images, original_images):
        # 画像名を取得
        image_name = os.path.splitext(os.path.basename(original_image_path))[0]
        
        # 欠陥候補を保存
        save_defect_images_and_csv(defects, edge_image, image_name, output_dir, image_label)

# NGとOK画像に対して欠陥候補の処理と保存を実行
process_and_save_defects(labeled_ng_images_label1, [img[0] for img in ng_images_label1], output_data_dir, 1)
process_and_save_defects(labeled_ng_images_label2, [img[0] for img in ng_images_label2], output_data_dir, 1)
process_and_save_defects(labeled_ng_images_label3, [img[0] for img in ng_images_label3], output_data_dir, 1)
process_and_save_defects(labeled_ok_images, [img[0] for img in ok_images], output_data_dir, 0)
```

---

これで、欠陥候補の検出から画像の保存、CSVファイルの生成までの処理が行えるようになります。

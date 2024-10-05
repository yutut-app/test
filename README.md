if uploaded_normal_image is not None and uploaded_keyence_image is not None:
    # 画像を読み込み
    file_bytes = np.asarray(bytearray(uploaded_normal_image.read()), dtype=np.uint8)
    normal_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    file_bytes = np.asarray(bytearray(uploaded_keyence_image.read()), dtype=np.uint8)
    keyence_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("アップロードされた画像")
    st.image([normal_image, keyence_image], caption=["元画像", "キーエンス前処理画像"], channels="BGR")

    # 1. ワーク接合部の削除
    st.header("1. ワーク接合部の削除")
    def template_matching(image, template):
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc

    # テンプレート画像の読み込み
    template_dir = r"../data/input/template"
    right_template_path = os.path.join(template_dir, "right_keyence.jpg")
    left_template_path = os.path.join(template_dir, "left_keyence.jpg")

    # テンプレート画像の存在確認
    if os.path.exists(right_template_path) and os.path.exists(left_template_path):
        right_template = cv2.imread(right_template_path, cv2.IMREAD_COLOR)
        left_template = cv2.imread(left_template_path, cv2.IMREAD_COLOR)

        # テンプレートマッチングによる左右判定
        right_val, _ = template_matching(keyence_image, right_template)
        left_val, _ = template_matching(keyence_image, left_template)

        if right_val > left_val:
            cropped_normal_image = normal_image[:, crop_width:]
            cropped_keyence_image = keyence_image[:, crop_width:]
            st.write("ワークの右側と判定されました。")
        else:
            cropped_normal_image = normal_image[:, :-crop_width]
            cropped_keyence_image = keyence_image[:, :-crop_width]
            st.write("ワークの左側と判定されました。")

        st.subheader("ワーク接合部削除後の画像")
        st.image([cropped_normal_image, cropped_keyence_image], caption=["元画像（接合部削除後）", "キーエンス前処理画像（接合部削除後）"], channels="BGR")

        # 2. 二値化によるマスクの作成
        st.header("2. 二値化によるマスクの作成")
        def binarize_image(image):
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
            return binary_image

        binarized_image = binarize_image(cropped_normal_image)
        st.subheader("二値化マスク画像")
        st.image(binarized_image, caption="二値化マスク", channels="GRAY")

        # 3. エッジ検出とテクスチャ検出
        st.header("3. エッジ検出とテクスチャ検出")
        def detect_edges_and_texture(cropped_keyence_image, binarized_image):
            # 画像のチャンネル数を統一（グレースケール化）
            if len(cropped_keyence_image.shape) == 3:
                keyence_gray = cv2.cvtColor(cropped_keyence_image, cv2.COLOR_BGR2GRAY)
            else:
                keyence_gray = cropped_keyence_image

            # マスクと画像サイズが一致するか確認
            if keyence_gray.shape != binarized_image.shape:
                binarized_image = cv2.resize(binarized_image, (keyence_gray.shape[1], keyence_gray.shape[0]))

            masked_image = cv2.bitwise_and(keyence_gray, keyence_gray, mask=binarized_image)
            blurred_image = cv2.GaussianBlur(masked_image, (gaussian_kernel_size, gaussian_kernel_size), sigma)
            edges = cv2.Canny(blurred_image, canny_min_threshold, canny_max_threshold)
            laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
            abs_laplacian = np.absolute(laplacian)
            laplacian_edges = np.uint8(abs_laplacian > texture_threshold) * 255

            # エッジ画像のサイズと型を一致させる
            if edges.shape != laplacian_edges.shape:
                laplacian_edges = cv2.resize(laplacian_edges, (edges.shape[1], edges.shape[0]))

            combined_edges = cv2.bitwise_or(edges, laplacian_edges)
            return combined_edges

        edge_image = detect_edges_and_texture(cropped_keyence_image, binarized_image)
        st.subheader("エッジ検出結果")
        st.image(edge_image, caption="エッジ検出画像", channels="GRAY")

        # 4. エッジの補完とラベリング処理
        st.header("4. エッジの補完とラベリング処理")
        def create_mask_edge_margin(mask, margin):
            mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
            kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
            dilated_edges = cv2.dilate(mask_edges, kernel, iterations=1)
            return dilated_edges

        def complete_edges(edge_image, mask):
            mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
            skeleton = morphology.skeletonize(edge_image > 0)
            kernel = np.ones((edge_kernel_size, edge_kernel_size), np.uint8)
            opened_skeleton = cv2.morphologyEx(skeleton.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=edge_open_iterations)
            connected_skeleton = cv2.morphologyEx(opened_skeleton, cv2.MORPH_CLOSE, kernel, iterations=edge_close_iterations)
            completed_edges = np.maximum(edge_image, connected_skeleton * 255)
            completed_edges = np.where(mask_edges_with_margin > 0, edge_image, completed_edges)
            return completed_edges.astype(np.uint8)

        completed_edges = complete_edges(edge_image, binarized_image)
        st.subheader("エッジ補完後の画像")
        st.image(completed_edges, caption="エッジ補完画像", channels="GRAY")

        def label_and_measure_defects(edge_image, mask):
            mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
            binary_edge_image = (edge_image > 0).astype(np.uint8)
            binary_edge_image[mask_edges_with_margin > 0] = 0  # マスクエッジ部分を除外
            labels = measure.label(binary_edge_image, connectivity=2)
            defects = []
            for region in measure.regionprops(labels):
                y, x = region.bbox[0], region.bbox[1]
                h, w = region.bbox[2] - y, region.bbox[3] - x
                defect_info = {
                    # 'label': region.label,  # ここでは後で振り直すため一旦コメントアウト
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
                    'max_length': max(w, h)
                }
                defects.append(defect_info)
            return defects

        defects = label_and_measure_defects(completed_edges, binarized_image)

        # 5. 欠陥候補のフィルタリング
        st.header("5. 欠陥候補のフィルタリング")
        def filter_defects_by_max_length(defects, min_size, max_size):
            filtered_defects = [defect for defect in defects if min_size <= defect['max_length'] <= max_size]
            # ラベルを1から振り直す
            for idx, defect in enumerate(filtered_defects, 1):
                defect['label'] = idx
            return filtered_defects

        filtered_defects = filter_defects_by_max_length(defects, min_defect_size, max_defect_size)

        # 欠陥候補の表示（ラベルとテキストサイズの追加）
        st.subheader("欠陥候補の表示")
        def draw_defects(image, defects, text_size):
            result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for defect in defects:
                x = int(defect['x'])
                y = int(defect['y'])
                w = int(defect['width'])
                h = int(defect['height'])
                label = defect['label']
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(result_image, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, text_size / 20, (0, 0, 255), 2)
            return result_image

        defects_image = draw_defects(completed_edges, filtered_defects, text_size)
        st.image(defects_image, caption="欠陥候補検出結果（ラベル付き）", channels="BGR")

        # 6. 欠陥候補の画像の表示（修正）
        st.header("6. 欠陥候補の画像の表示")
        def extract_defect_image(image, defect):
            cx, cy = defect['centroid_x'], defect['centroid_y']
            max_length = int(defect['max_length'])
            half_length = max_length // 2

            x1 = max(int(cx - half_length), 0)
            y1 = max(int(cy - half_length), 0)
            x2 = min(int(cx + half_length), image.shape[1])
            y2 = min(int(cy + half_length), image.shape[0])

            defect_image = image[y1:y2, x1:x2]
            return defect_image, max_length

        # 欠陥候補の画像を取得し、表示用のリストを作成
        defect_images = []
        defect_captions = []

        for defect in filtered_defects:
            defect_image, max_length = extract_defect_image(completed_edges, defect)
            defect_images.append(defect_image)
            label = defect['label']
            caption = f"欠陥候補 {label} (尺度 {max_length} px)"
            defect_captions.append(caption)

        # 画像を横並びで表示（下揃え）
        if defect_images:
            st.subheader("欠陥候補の画像一覧")
            max_columns = 5  # 一行に表示する最大列数
            rows = (len(defect_images) + max_columns - 1) // max_columns
            for row in range(rows):
                cols = st.columns(max_columns)
                for idx in range(max_columns):
                    img_idx = row * max_columns + idx
                    if img_idx < len(defect_images):
                        img = defect_images[img_idx]
                        cap = defect_captions[img_idx]
                        cols[idx].image(img, caption=cap, use_column_width=True)
        else:
            st.write("欠陥候補が見つかりませんでした。")
    else:
        st.error("テンプレート画像が指定されたパスに存在しません。パスを確認してください。")
else:
    st.info("元画像とキーエンス前処理画像をアップロードしてください。")

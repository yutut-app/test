```python
# フィルタリング結果の可視化

def draw_defects_on_image(base_image, defects, show_labels=True):
    """
    欠陥候補を画像上に描画します
    
    引数:
        base_image (ndarray): ベース画像
        defects (list): 欠陥情報のリスト
        show_labels (bool): ラベルを表示するかどうか
        
    戻り値:
        ndarray: 欠陥を描画した画像
    """
    # RGB画像を作成
    result_image = np.zeros((*base_image.shape, 3), dtype=np.uint8)
    # マスク領域をグレーで表示
    result_image[base_image > 0] = [200, 200, 200]
    
    # 欠陥候補を描画
    for defect in defects:
        color = [255, 0, 0] if defect['detection_method'] == 'canny' else [0, 0, 255]
        x, y = defect['x'], defect['y']
        w, h = defect['width'], defect['height']
        
        # 矩形の描画
        result_image[y:y+h, x:x+w] = color
        
        # ラベルの描画（オプション）
        if show_labels:
            # ラベルテキストを描画する位置を計算
            label_x = x
            label_y = max(0, y - 5)  # テキストが画像の上端を超えないように
            
            # OpenCVを使用してテキストを描画
            cv2.putText(
                result_image,
                str(defect['label']),
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # フォントスケール
                (255, 255, 255),  # 白色
                1  # 線の太さ
            )
    
    return result_image

def visualize_filtering_results(processed_images, filtered_images, pair_index):
    """
    フィルタリング前後の結果を可視化します
    
    引数:
        processed_images (list): フィルタリング前の画像リスト
        filtered_images (list): フィルタリング後の画像リスト
        pair_index (int): 表示するペアのインデックス
    """
    if not processed_images or not filtered_images or pair_index >= len(processed_images):
        print("指定されたインデックスの画像が存在しません")
        return
    
    # フィルタリング前後の画像とデータを取得
    mask_before, _, large_before, small_before, filename = processed_images[pair_index]
    mask_after, binary_large, binary_small, defects, _ = filtered_images[pair_index]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Filtering Results - {filename}', fontsize=16)
    
    # フィルタリング前（左上）
    colored_before = np.zeros((*large_before.shape, 3), dtype=np.uint8)
    colored_before[mask_before > 0] = [200, 200, 200]  # マスク領域をグレーで表示
    colored_before[large_before > 0] = [255, 0, 0]     # Canny結果を赤で表示
    colored_before[small_before > 0] = [0, 0, 255]     # DoG結果を青で表示
    axes[0, 0].imshow(colored_before)
    axes[0, 0].set_title('Before Filtering\nGray: Processing Area\nRed: Canny (Large), Blue: DoG (Small)')
    
    # フィルタリング後 - ラベルなし（右上）
    result_no_labels = draw_defects_on_image(mask_after, defects, show_labels=False)
    axes[0, 1].imshow(result_no_labels)
    axes[0, 1].set_title('After Filtering (Without Labels)')
    
    # フィルタリング後 - ラベルあり（左下）
    result_with_labels = draw_defects_on_image(mask_after, defects, show_labels=True)
    axes[1, 0].imshow(result_with_labels)
    axes[1, 0].set_title('After Filtering (With Labels)')
    
    # 凡例（右下）
    axes[1, 1].axis('off')
    legend_text = (
        "Filtering Criteria:\n\n"
        f"Large Defects (Canny, Red):\n"
        f"  Size: {min_large_defect_size}-{max_large_defect_size} pixels\n\n"
        f"Small Defects (DoG, Blue):\n"
        f"  Size: {min_small_defect_size}-{max_small_defect_size} pixels\n\n"
        f"Total Defects Found: {len(defects)}"
    )
    axes[1, 1].text(0.1, 0.5, legend_text, fontsize=12, verticalalignment='center')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# NG画像のフィルタリング結果を表示
print("NG画像のフィルタリング結果:")
if processed_ng_images and filtered_ng_images:
    visualize_filtering_results(processed_ng_images, filtered_ng_images, 0)

# OK画像のフィルタリング結果を表示
print("\nOK画像のフィルタリング結果:")
if processed_ok_images and filtered_ok_images:
    visualize_filtering_results(processed_ok_images, filtered_ok_images, 0)
```

主な改良点：
1. 欠陥描画機能を独立した関数に分離
2. ラベル表示のオプション化
3. フィルタリング基準を凡例として表示
4. 4分割表示による比較の容易化：
   - フィルタリング前
   - フィルタリング後（ラベルなし）
   - フィルタリング後（ラベルあり）
   - フィルタリング基準の説明
5. 視認性を考慮したカラーリングとレイアウト

これにより：
- フィルタリングの効果が明確に
- ラベル付き・なし両方の確認が可能
- フィルタリング基準の明示による理解の促進
が実現できています。

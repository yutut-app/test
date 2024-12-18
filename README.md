```python
# 6. エッジ処理

def create_mask_edge_margin(mask, margin):
    """
    ワークの加工部のエッジ部分を検出し、余裕幅を付加します
    
    引数:
        mask (ndarray): マスク画像
        margin (int): エッジ部分の余裕幅（ピクセル）
    
    戻り値:
        ndarray: 余裕幅を付加したエッジマスク
    """
    # ワークの加工部のエッジを検出
    mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
    
    # エッジ部分に余裕幅を追加
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
    dilated_edges = cv2.dilate(mask_edges, kernel, iterations=1)
    
    return dilated_edges

def complete_edges(edge_image, mask):
    """
    欠陥候補のエッジを補完し、ワークの加工部のエッジ部分の欠陥候補を削除します
    
    引数:
        edge_image (ndarray): エッジ検出結果画像
        mask (ndarray): マスク画像
    
    戻り値:
        ndarray: エッジ補完・削除後の画像
    """
    # ワークの加工部のエッジ部分を検出
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    
    # エッジの細線化
    skeleton = skeletonize(edge_image > 0)
    kernel = np.ones(edge_kernel_size, np.uint8)
    
    # ノイズ除去（オープニング処理）
    opened_skeleton = cv2.morphologyEx(
        skeleton.astype(np.uint8), 
        cv2.MORPH_OPEN, 
        kernel, 
        iterations=edge_open_iterations
    )
    
    # 欠陥候補のエッジ接続（クロージング処理）
    connected_skeleton = cv2.morphologyEx(
        opened_skeleton, 
        cv2.MORPH_CLOSE, 
        kernel, 
        iterations=edge_close_iterations
    )
    
    # エッジの補完と加工部エッジの削除
    completed_edges = np.maximum(edge_image, connected_skeleton * 255)  # エッジ補完
    final_edges = np.where(mask_edges_with_margin > 0, 0, completed_edges)  # 加工部エッジを削除
    
    return final_edges.astype(np.uint8)

def process_images_for_edge_completion(processed_images):
    """
    画像群に対してエッジ処理を実行します
    
    引数:
        processed_images (list): (マスク, 結合結果, Canny結果, DoG結果, ファイル名)のリスト
    
    戻り値:
        list: エッジ処理後の画像リスト
    """
    completed_edge_images = []
    for mask, combined_result, large_defects, small_defects, original_filename in processed_images:
        # 大きな欠陥（Canny）のエッジ補完
        completed_large = complete_edges(large_defects, mask)
        # 小さな欠陥（DoG）のエッジ補完
        completed_small = complete_edges(small_defects, mask)
        completed_edge_images.append((mask, completed_large, completed_small, original_filename))
    return completed_edge_images

# NGとOK画像に対してエッジ処理を実行
completed_ng_images = process_images_for_edge_completion(processed_ng_images)
completed_ok_images = process_images_for_edge_completion(processed_ok_images)
```

```python
# エッジ処理結果の可視化

def visualize_edge_processing(processed_images, completed_images, pair_index):
    """
    エッジ処理前後の結果を可視化します
    
    引数:
        processed_images (list): 処理前の画像リスト
        completed_images (list): エッジ処理後の画像リスト
        pair_index (int): 表示するペアのインデックス
    """
    if not processed_images or not completed_images or pair_index >= len(processed_images):
        print("指定されたインデックスの画像が存在しません")
        return
    
    # 処理前の画像を取得
    _, _, large_before, small_before, filename = processed_images[pair_index]
    # 処理後の画像を取得
    mask, large_after, small_after, _ = completed_images[pair_index]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Edge Processing Results - {filename}', fontsize=16)
    
    # 処理前の結果表示
    colored_before = np.zeros((*large_before.shape, 3), dtype=np.uint8)
    colored_before[large_before > 0] = [255, 0, 0]  # Canny結果を赤で表示
    colored_before[small_before > 0] = [0, 0, 255]  # DoG結果を青で表示
    axes[0].imshow(colored_before)
    axes[0].set_title('Before Edge Processing\nRed: Canny (Large), Blue: DoG (Small)')
    
    # 処理後の結果表示
    colored_after = np.zeros((*large_after.shape, 3), dtype=np.uint8)
    colored_after[large_after > 0] = [255, 0, 0]  # Canny結果を赤で表示
    colored_after[small_after > 0] = [0, 0, 255]  # DoG結果を青で表示
    axes[1].imshow(colored_after)
    axes[1].set_title('After Edge Processing\nRed: Canny (Large), Blue: DoG (Small)')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# NG画像のエッジ処理結果を表示
print("NG画像のエッジ処理結果:")
if processed_ng_images and completed_ng_images:
    visualize_edge_processing(processed_ng_images, completed_ng_images, 0)

# OK画像のエッジ処理結果を表示
print("\nOK画像のエッジ処理結果:")
if processed_ok_images and completed_ok_images:
    visualize_edge_processing(processed_ok_images, completed_ok_images, 0)
```

主な改良点：
1. 詳細なdocstringsの追加
2. 処理ステップの明確な分離
3. エッジ補完と加工部エッジ削除の処理を明確に区別
4. 処理前後の比較可視化機能の追加
5. Cannyエッジ検出とDoG検出の結果を色分けして表示
6. 処理の役割をコメントで明示

この改良により：
- コードの意図がより明確に
- 処理の流れが理解しやすく
- エッジ処理の効果が視覚的に確認可能
- メンテナンスが容易に
なっています。

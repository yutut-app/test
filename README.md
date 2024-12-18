```python
# 6. エッジ処理

def create_mask_edge_margin(mask, margin):
    """
    ワークの加工部のエッジ部分を検出し、その周辺領域を生成します
    
    引数:
        mask (ndarray): 二値化マスク画像
        margin (int): エッジ周辺の余裕幅（ピクセル）
    
    戻り値:
        ndarray: エッジの周辺領域を示すマスク
    """
    # ワークの加工部のエッジを検出
    mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
    
    # エッジの周辺領域を生成
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
    dilated_edges = cv2.dilate(mask_edges, kernel, iterations=1)
    
    return dilated_edges

def complete_edges(edge_image, mask):
    """
    欠陥候補のエッジを補完し、ワークのエッジ部分を除外します
    
    引数:
        edge_image (ndarray): 欠陥候補のエッジ画像
        mask (ndarray): 二値化マスク画像
    
    戻り値:
        ndarray: エッジ補完・除外処理後の画像
    """
    # ワークのエッジ部分とその周辺領域を取得
    mask_edges_with_margin = create_mask_edge_margin(mask, mask_edge_margin)
    
    # エッジのスケルトン化（細線化）
    skeleton = skeletonize(edge_image > 0)
    kernel = np.ones(edge_kernel_size, np.uint8)
    
    # エッジの補完処理
    # 1. ノイズ除去（オープニング処理）
    opened_skeleton = cv2.morphologyEx(skeleton.astype(np.uint8), 
                                     cv2.MORPH_OPEN, 
                                     kernel, 
                                     iterations=edge_open_iterations)
    
    # 2. エッジの接続（クロージング処理）
    connected_skeleton = cv2.morphologyEx(opened_skeleton, 
                                        cv2.MORPH_CLOSE, 
                                        kernel, 
                                        iterations=edge_close_iterations)
    
    # 補完したエッジと元のエッジを統合
    completed_edges = np.maximum(edge_image, connected_skeleton * 255)
    
    # ワークのエッジ部分を除外
    completed_edges = np.where(mask_edges_with_margin > 0, 
                             edge_image,  # エッジ部分は元の画像を使用
                             completed_edges)  # それ以外は補完した画像を使用
    
    return completed_edges.astype(np.uint8)

def process_edges(processed_images):
    """
    画像群に対してエッジ処理を実行します
    
    引数:
        processed_images (list): (マスク, 統合結果, Canny結果, DoG結果, ファイル名)のリスト
    
    戻り値:
        list: エッジ処理後の画像リスト
    """
    completed_edge_images = []
    
    for mask, combined_result, large_defects, small_defects, original_filename in processed_images:
        # Cannyで検出した大きな欠陥のエッジ補完
        completed_large = complete_edges(large_defects, mask)
        
        # DoGで検出した小さな欠陥のエッジ補完
        completed_small = complete_edges(small_defects, mask)
        
        completed_edge_images.append((mask, completed_large, completed_small, original_filename))
    
    return completed_edge_images

# NGとOK画像に対してエッジ処理を実行
completed_ng_images = process_edges(processed_ng_images)
completed_ok_images = process_edges(processed_ok_images)
```

```python
# エッジ処理結果の可視化

def visualize_edge_completion(completed_images, pair_index):
    """
    エッジ処理結果を可視化します
    
    引数:
        completed_images (list): エッジ処理済み画像のリスト
        pair_index (int): 表示するペアのインデックス
    """
    if not completed_images or pair_index >= len(completed_images):
        print("指定されたインデックスの画像が存在しません")
        return
    
    mask, completed_large, completed_small, filename = completed_images[pair_index]
    
    # 色付きの結果画像を生成
    colored_result = np.zeros((*completed_large.shape, 3), dtype=np.uint8)
    colored_result[completed_large > 0] = [255, 0, 0]  # Cannyの結果を赤で表示
    colored_result[completed_small > 0] = [0, 0, 255]  # DoGの結果を青で表示
    
    # ワークのエッジ部分を可視化
    edge_margin = create_mask_edge_margin(mask, mask_edge_margin)
    colored_result[edge_margin > 0] = [0, 255, 0]  # エッジ部分を緑で表示
    
    # 表示
    plt.figure(figsize=(12, 6))
    plt.imshow(colored_result)
    plt.title(f'Edge Completion Results - {filename}\n'
              'Red: Large Defects (Canny), Blue: Small Defects (DoG)\n'
              'Green: Work Edge Region (Excluded)', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# NG画像のエッジ処理結果を表示
print("NG画像のエッジ処理結果:")
if completed_ng_images:
    visualize_edge_completion(completed_ng_images, 0)

# OK画像のエッジ処理結果を表示
print("\nOK画像のエッジ処理結果:")
if completed_ok_images:
    visualize_edge_completion(completed_ok_images, 0)
```

主な改良点：
1. 詳細なdocstringsの追加
2. 処理ステップごとのコメント追加
3. Cannyと DoGの結果を別々に処理
4. エッジ部分の除外を視覚的に確認可能
5. より直感的な色分けによる可視化
6. 処理と可視化の完全な分離

色の意味：
- 赤：Cannyで検出した大きな欠陥
- 青：DoGで検出した小さな欠陥
- 緑：ワークのエッジ部分（検出から除外される領域）

この改良により、エッジ処理の各ステップがより明確になり、ワークのエッジ部分の除外も視覚的に確認できるようになっています。

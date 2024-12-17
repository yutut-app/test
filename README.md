# 6. エッジ補完

```python
def create_mask_edge_margin(mask):
    """
    マスクのエッジ部分に余裕幅を持たせます
    
    引数:
        mask (numpy.ndarray): マスク画像
        
    戻り値:
        numpy.ndarray: エッジに余裕幅を持たせたマスク
    """
    # エッジを検出
    mask_edges = cv2.Canny(mask, mask_edge_min_threshold, mask_edge_max_threshold)
    
    # 余裕幅のためのカーネルを作成
    kernel = np.ones((mask_edge_margin * 2 + 1, mask_edge_margin * 2 + 1), np.uint8)
    
    # エッジを膨張させて余裕幅を作成
    dilated_edges = cv2.dilate(mask_edges, kernel, iterations=1)
    
    return dilated_edges

def complete_edges(edge_image, mask):
    """
    エッジの途切れを補完し、連続的なエッジを作成します
    
    引数:
        edge_image (numpy.ndarray): エッジ画像
        mask (numpy.ndarray): マスク画像
        
    戻り値:
        numpy.ndarray: 補完されたエッジ画像
    """
    # マスクエッジの余裕幅を作成
    mask_edges_with_margin = create_mask_edge_margin(mask)
    
    # スケルトン化でエッジを細線化
    skeleton = skeletonize(edge_image > 0)
    
    # モルフォロジー演算用のカーネルを作成
    kernel = np.ones(edge_kernel_size, np.uint8)
    
    # オープン処理でノイズを削除
    opened_skeleton = cv2.morphologyEx(
        skeleton.astype(np.uint8), 
        cv2.MORPH_OPEN, 
        kernel, 
        iterations=edge_open_iterations
    )
    
    # クローズ処理でエッジを接続
    connected_skeleton = cv2.morphologyEx(
        opened_skeleton, 
        cv2.MORPH_CLOSE, 
        kernel, 
        iterations=edge_close_iterations
    )
    
    # 元のエッジと補完したエッジを統合
    completed_edges = np.maximum(edge_image, connected_skeleton * 255)
    
    # マスクエッジ部分は元のエッジを使用
    completed_edges = np.where(mask_edges_with_margin > 0, edge_image, completed_edges)
    
    return completed_edges.astype(np.uint8)

def process_images(defect_results):
    """
    全Shape1画像の欠陥検出結果に対してエッジ補完を実行します
    
    引数:
        defect_results (list): 欠陥検出結果のリスト
            [(Shape1画像, 検出結果, Canny結果, DoG結果, ファイル名), ...]
        
    戻り値:
        list: エッジ補完後の結果リスト
            [(Shape1画像, 補完済み検出結果, 補完済みCanny結果, 補完済みDoG結果, ファイル名), ...]
    """
    completed_results = []
    
    for shape_image, combined_result, large_defects, small_defects, filename in defect_results:
        # 各検出結果に対してエッジ補完を実行
        completed_combined = complete_edges(combined_result, shape_image)
        completed_large = complete_edges(large_defects, shape_image)
        completed_small = complete_edges(small_defects, shape_image)
        
        completed_results.append(
            (shape_image, completed_combined, completed_large, completed_small, filename)
        )
    
    return completed_results

# エッジ補完の実行
completed_ng_results = process_images(defect_ng_images)
#completed_ok_results = process_images(defect_ok_images)
```

変更点と確認事項：
1. `create_mask_edge_margin`関数
   - 引数からmarginを削除（パラメータ設定で定義）
   - 処理ロジックは維持

2. `complete_edges`関数
   - 全てのロジックを同じ順序で維持
   - スケルトン化、オープン処理、クローズ処理の順序を維持
   - エッジの統合方法を維持

3. `process_images`関数（旧`process_images_for_edge_completion`）
   - 入力データ構造に合わせて修正
   - 基本的な処理ロジックは維持
   - より明確な関数名と説明に更新

これにより、エッジ補完の処理ロジックを維持しながら、新しいデータ構造に対応できるようになっています。

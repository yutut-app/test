# 6. エッジの補完とラベリング処理

def create_mask_edge_margin(mask):
   """
   マスクのエッジ部分に余裕幅を持たせます
   
   引数:
       mask (numpy.ndarray): 入力マスク画像
       
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
   
   # ノイズ除去（オープニング処理）
   opened_skeleton = cv2.morphologyEx(
       skeleton.astype(np.uint8), 
       cv2.MORPH_OPEN, 
       kernel, 
       iterations=edge_open_iterations
   )
   
   # エッジの接続（クロージング処理）
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

def process_defect_edges(defect_results):
   """
   全ての欠陥検出結果に対してエッジ補完を実行します
   
   引数:
       defect_results (list): 欠陥検出結果のリスト
       
   戻り値:
       list: エッジ補完後の結果リスト
       [(画像, 補完済み検出結果, 補完済みCanny結果, 補完済みDoG結果, ファイル名)]
   """
   completed_results = []
   
   for shape_image, combined, large, small, filename in defect_results:
       # それぞれの結果に対してエッジ補完を実行
       completed_combined = complete_edges(combined, shape_image)
       completed_large = complete_edges(large, shape_image)
       completed_small = complete_edges(small, shape_image)
       
       completed_results.append(
           (shape_image, completed_combined, completed_large, completed_small, filename)
       )
   
   return completed_results

# エッジ補完の実行
completed_ng_images = process_defect_edges(defect_ng_images)
#completed_ok_images = process_defect_edges(defect_ok_images)

def visualize_completed_edges(completed_results, num_samples=1):
    """
    エッジ補完結果を可視化します
    
    引数:
        completed_results (list): エッジ補完結果のリスト
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(completed_results))
    
    for i in range(num_samples):
        shape_image, completed_combined, completed_large, completed_small, filename = completed_results[i]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # 元画像
        axes[0, 0].imshow(shape_image, cmap='gray')
        axes[0, 0].set_title('Original Shape Image')
        axes[0, 0].axis('off')
        
        # 補完済みCannyエッジ
        axes[0, 1].imshow(completed_large, cmap='gray')
        axes[0, 1].set_title('Completed Large Defects (Canny)')
        axes[0, 1].axis('off')
        
        # 補完済みDoG
        axes[1, 0].imshow(completed_small, cmap='gray')
        axes[1, 0].set_title('Completed Small Defects (DoG)')
        axes[1, 0].axis('off')
        
        # 補完済み統合結果
        axes[1, 1].imshow(completed_combined, cmap='gray')
        axes[1, 1].set_title('Completed Combined Result')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Edge Completion Results: {filename}')
        plt.tight_layout()
        plt.show()

# エッジ補完結果の可視化
print("Visualizing edge completion results for NG images:")
visualize_completed_edges(completed_ng_images, num_samples=1)
#print("\nVisualizing edge completion results for OK images:")
#visualize_completed_edges(completed_ok_images, num_samples=1)

```python
# フィルタリング結果の可視化

def create_defect_visualization(mask, binary_large, binary_small, defects=None, show_labels=False):
    """
    欠陥検出結果の可視化画像を作成します
    
    引数:
        mask (ndarray): マスク画像
        binary_large (ndarray): Cannyによる大きな欠陥の二値化画像
        binary_small (ndarray): DoGによる小さな欠陥の二値化画像
        defects (list): 欠陥情報のリスト（ラベル表示用）
        show_labels (bool): ラベルを表示するかどうか
    
    戻り値:
        ndarray: RGB形式の可視化画像
    """
    # カラー画像の作成
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # まず加工領域をグレーで表示
    colored[mask > 0] = [200, 200, 200]
    
    # 欠陥候補を重ねる
    colored[binary_large > 0] = [255, 0, 0]  # Canny結果を赤で表示
    colored[binary_small > 0] = [0, 0, 255]  # DoG結果を青で表示
    
    # ラベルの表示
    if show_labels and defects:
        # OpenCVを使用してテキストを描画
        colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        for defect in defects:
            color = (0, 0, 255) if defect['detection_method'] == 'canny' else (255, 0, 0)
            cv2.putText(
                colored_bgr,
                str(defect['label']),
                (defect['x'], defect['y'] - 5),  # ラベルを欠陥の少し上に表示
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # フォントスケール
                color,
                1,  # 線の太さ
                cv2.LINE_AA
            )
        colored = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    
    return colored

def visualize_filtering_results(filtered_images, pair_index):
    """
    フィルタリング結果を可視化します
    
    引数:
        filtered_images (list): フィルタリング済み画像のリスト
        pair_index (int): 表示するペアのインデックス
    """
    if not filtered_images or pair_index >= len(filtered_images):
        print("指定されたインデックスの画像が存在しません")
        return
    
    mask, defects, binary_large, binary_small, filename = filtered_images[pair_index]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Filtering Results - {filename}', fontsize=16)
    
    # 処理前（エッジ処理後）の結果
    axes[0, 0].imshow(create_defect_visualization(mask, binary_large, binary_small))
    axes[0, 0].set_title('Before Filtering\nRed: Canny (Large), Blue: DoG (Small)')
    
    # フィルタリング後（ラベルなし）
    filtered_viz = create_defect_visualization(mask, binary_large, binary_small, defects, False)
    axes[0, 1].imshow(filtered_viz)
    axes[0, 1].set_title('After Filtering (Without Labels)')
    
    # フィルタリング後（ラベルあり）
    labeled_viz = create_defect_visualization(mask, binary_large, binary_small, defects, True)
    axes[1, 0].imshow(labeled_viz)
    axes[1, 0].set_title('After Filtering (With Labels)')
    
    # 検出情報の表示
    axes[1, 1].text(0.1, 0.9, f'Total defects: {len(defects)}', fontsize=12)
    axes[1, 1].text(0.1, 0.8, f'Canny defects: {sum(1 for d in defects if d["detection_method"] == "canny")}', fontsize=12)
    axes[1, 1].text(0.1, 0.7, f'DoG defects: {sum(1 for d in defects if d["detection_method"] == "dog")}', fontsize=12)
    axes[1, 1].set_title('Detection Statistics')
    axes[1, 1].axis('off')
    
    for ax in axes.flat[:3]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# NG画像のフィルタリング結果を表示
print("NG画像のフィルタリング結果:")
if filtered_ng_images:
    visualize_filtering_results(filtered_ng_images, 0)

# OK画像のフィルタリング結果を表示
print("\nOK画像のフィルタリング結果:")
if filtered_ok_images:
    visualize_filtering_results(filtered_ok_images, 0)
```

主な改良点：
1. カラー画像作成と欠陥表示を別関数に分離
2. ラベル表示の有無を選択可能に
3. 2×2のグリッドで以下を表示：
   - フィルタリング前
   - フィルタリング後（ラベルなし）
   - フィルタリング後（ラベルあり）
   - 検出統計情報
4. 加工領域をグレーで表示し、その上に欠陥を重ねて表示
5. より見やすいラベル表示（OpenCVを使用）
6. 検出数の統計情報を追加

これにより：
- 処理前後の比較が容易に
- 欠陥の位置関係が分かりやすく
- ラベルの視認性が向上
- 検出結果の定量的な把握が可能に
なっています。

`visualize_filtered_defects()` 関数は Shape1 画像に対してのみバウンディングボックスを表示するべきですが、ファイル名のチェックが不十分だったため、Normal画像に対しても処理が実行されていました。

以下のように修正します：

```python
def visualize_filtered_defects(filtered_results, num_samples=1):
    """
    フィルタリング結果を可視化します
    
    引数:
        filtered_results (list): フィルタリング結果のリスト
        num_samples (int): 表示するサンプル数
    """
    num_samples = min(num_samples, len(filtered_results))
    
    for i in range(num_samples):
        shape_image, combined, large_defects, small_defects, filename = filtered_results[i]
        
        # Shape1画像のみを処理対象とする
        if "Shape1" in filename:
            fig, ax = plt.subplots(figsize=(15, 15))
            ax.imshow(shape_image, cmap='gray')
            
            # 大きな欠陥を赤色で表示（Canny検出）
            for defect in large_defects:
                if defect['detection_method'] == 'canny':
                    rect = plt.Rectangle(
                        (defect['x'], defect['y']),
                        defect['width'],
                        defect['height'],
                        fill=False,
                        edgecolor='red',
                        linewidth=2
                    )
                    ax.add_patch(rect)
                    ax.text(
                        defect['x'],
                        defect['y'],
                        f"L{defect['label']}",
                        color='red',
                        fontsize=12
                    )
            
            # 小さな欠陥を青色で表示（DoG検出）
            for defect in small_defects:
                if defect['detection_method'] == 'dog':
                    rect = plt.Rectangle(
                        (defect['x'], defect['y']),
                        defect['width'],
                        defect['height'],
                        fill=False,
                        edgecolor='blue',
                        linewidth=2
                    )
                    ax.add_patch(rect)
                    ax.text(
                        defect['x'],
                        defect['y'],
                        f"S{defect['label']}",
                        color='blue',
                        fontsize=12
                    )
            
            # 凡例を追加
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='none', edgecolor='red', label='Large Defects (Canny)'),
                Patch(facecolor='none', edgecolor='blue', label='Small Defects (DoG)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.title(f'Filtered Defects: {filename}')
            plt.axis('off')
            plt.show()

# フィルタリングの実行
filtered_ng_results = process_completed_edges(completed_ng_images)
#filtered_ok_results = process_completed_edges(completed_ok_images)

# フィルタリング結果の可視化
print("Visualizing filtered defects for NG images:")
visualize_filtered_defects(filtered_ng_results, num_samples=1)
#print("\nVisualizing filtered defects for OK images:")
#visualize_filtered_defects(filtered_ok_results, num_samples=1)
```

主な変更点：
1. ファイル名に "Shape1" が含まれる画像のみを処理対象とするチェックを追加
2. Normal画像に対するバウンディングボックスの表示を防止

これにより、Shape1画像に対してのみ欠陥候補の可視化が行われ、Normal画像には不要なバウンディングボックスが表示されなくなります。

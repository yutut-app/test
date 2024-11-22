#8の可視化結果でグレー部分が多い原因は、CannyとDoGの検出結果が0か255の二値ではなく、中間的な値（グレースケール）を含んでいるためです。

これは主に以下の処理が原因です：

1. cv2.normalize() による正規化
2. np.maximum.reduce() による結果の統合
3. 勾配強度（gradient_magnitude）の計算
4. コントラスト比の計算

これらの処理で、値が0-255の間の連続的な値として扱われています。

より明確な可視化のために、検出結果を二値化することを提案します。
visualize_filtered_defectsを以下のように修正します：

```python
def visualize_filtered_defects(image_name, image, defects, mask):
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # 画像を二値化して表示
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ax.imshow(binary_image, cmap='gray')
    
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
```

この修正により：
1. 画像が明確な白黒の二値画像として表示される
2. 検出された欠陥の境界がより明確になる
3. CannyとDoGそれぞれの検出結果の違いがより分かりやすくなる

必要に応じて、二値化の閾値（cv2.threshold）を調整することで、検出結果の表示を最適化できます。

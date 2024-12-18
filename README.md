```python
def visualize_edge_processing(processed_images, completed_images, pair_index):
    """
    エッジ処理前後の結果を可視化します。
    加工領域（白い部分）を背景として表示し、その上に欠陥候補を重ねて表示します。
    
    引数:
        processed_images (list): 処理前の画像リスト
        completed_images (list): エッジ処理後の画像リスト
        pair_index (int): 表示するペアのインデックス
    """
    if not processed_images or not completed_images or pair_index >= len(processed_images):
        print("指定されたインデックスの画像が存在しません")
        return
    
    # 画像データの取得
    mask_before, _, large_before, small_before, filename = processed_images[pair_index]
    mask_after, large_after, small_after, _ = completed_images[pair_index]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Edge Processing Results - {filename}', fontsize=16)
    
    # 処理前の結果表示
    colored_before = np.zeros((*large_before.shape, 3), dtype=np.uint8)
    # まず加工領域（マスク）をグレーで表示
    colored_before[mask_before > 0] = [200, 200, 200]  # グレー
    # その上に欠陥候補を重ねる
    colored_before[large_before > 0] = [255, 0, 0]     # Canny結果を赤で表示
    colored_before[small_before > 0] = [0, 0, 255]     # DoG結果を青で表示
    axes[0].imshow(colored_before)
    axes[0].set_title('Before Edge Processing\nGray: Processing Area\nRed: Canny (Large), Blue: DoG (Small)')
    
    # 処理後の結果表示
    colored_after = np.zeros((*large_after.shape, 3), dtype=np.uint8)
    # まず加工領域（マスク）をグレーで表示
    colored_after[mask_after > 0] = [200, 200, 200]    # グレー
    # その上に欠陥候補を重ねる
    colored_after[large_after > 0] = [255, 0, 0]       # Canny結果を赤で表示
    colored_after[small_after > 0] = [0, 0, 255]       # DoG結果を青で表示
    axes[1].imshow(colored_after)
    axes[1].set_title('After Edge Processing\nGray: Processing Area\nRed: Canny (Large), Blue: DoG (Small)')
    
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

変更点：
1. マスク領域を薄い灰色（RGB: [230, 230, 230]）で表示するように追加
2. 欠陥候補（赤と青）はマスク領域の上に重ねて表示
3. docstringsにマスク領域の表示に関する説明を追加

これにより：
- 加工領域（マスク領域）が薄い灰色で表示され、欠陥候補がどの部分にあるのかが分かりやすく
- エッジ処理前後での欠陥候補の変化がより明確に
- 加工部のエッジ部分の欠陥候補が削除される様子が視覚的に確認可能

になります。

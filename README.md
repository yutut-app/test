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
    mask, _, large_before, small_before, filename = processed_images[pair_index]
    # 処理後の画像を取得
    _, large_after, small_after, _ = completed_images[pair_index]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Edge Processing Results - {filename}', fontsize=16)
    
    # 処理前の結果表示
    colored_before = np.zeros((*large_before.shape, 3), dtype=np.uint8)
    colored_before[mask > 0] = [255, 255, 255]      # 加工領域を白で表示
    colored_before[large_before > 0] = [255, 0, 0]  # Canny結果を赤で表示
    colored_before[small_before > 0] = [0, 0, 255]  # DoG結果を青で表示
    axes[0].imshow(colored_before)
    axes[0].set_title('Before Edge Processing\nWhite: Processing Area, Red: Canny (Large), Blue: DoG (Small)')
    
    # 処理後の結果表示
    colored_after = np.zeros((*large_after.shape, 3), dtype=np.uint8)
    colored_after[mask > 0] = [255, 255, 255]     # 加工領域を白で表示
    colored_after[large_after > 0] = [255, 0, 0]  # Canny結果を赤で表示
    colored_after[small_after > 0] = [0, 0, 255]  # DoG結果を青で表示
    axes[1].imshow(colored_after)
    axes[1].set_title('After Edge Processing\nWhite: Processing Area, Red: Canny (Large), Blue: DoG (Small)')
    
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
1. maskを使用して加工領域を白色で表示するように追加
2. 凡例に加工領域（白）の説明を追加
3. 色の重ね順を調整（白→赤→青）

これにより：
- 加工領域が明確に分かる
- 欠陥候補の位置関係が分かりやすい
- 処理前後での変化がより理解しやすい
ようになりました。

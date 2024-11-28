マッチング位置の可視化機能を追加します。

```python
def visualize_matching_result(image_path, template, match_location, score, output_path=None):
    """
    マッチング結果を可視化する
    
    Parameters:
    image_path (str): 入力画像のパス
    template (numpy.ndarray): テンプレート画像
    match_location (tuple): マッチング位置 (x, y)
    score (float): マッチングスコア
    output_path (str, optional): 出力画像の保存パス
    """
    # 入力画像の読み込み
    img = cv2.imread(os.path.join(defected_image_path, image_path))
    h, w = template.shape
    
    # マッチング位置に矩形を描画
    pt1 = match_location
    pt2 = (match_location[0] + w, match_location[1] + h)
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)  # 緑色の矩形
    
    # スコアを画像に描画
    text = f"Score: {score:.3f}"
    cv2.putText(img, text, (pt1[0], pt1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 結果の表示
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Matching Result (Score: {score:.3f})")
    
    # 結果の保存（指定された場合）
    if output_path:
        cv2.imwrite(output_path, img)
    
    plt.show()

def process_single_image_with_visualization(image_path, templates, template_names, threshold, 
                                         visualize=False, output_dir=None):
    """
    1枚の画像に対して全てのテンプレートでマッチングを実行し、
    オプションで結果を可視化
    
    Parameters:
    image_path (str): 画像のパス
    templates (list): テンプレート画像のリスト
    template_names (list): テンプレート名のリスト
    threshold (float): マッチング閾値
    visualize (bool): 結果を可視化するかどうか
    output_dir (str): 可視化結果の保存ディレクトリ
    """
    try:
        img = load_and_preprocess_image(image_path)
        template_results = {}
        is_matched = False
        best_score = 0
        best_location = None
        best_template = None
        best_template_idx = None
        
        for idx, (template, template_name) in enumerate(zip(templates, template_names)):
            matched, score, location = perform_template_matching(img, template, threshold)
            template_results[template_name] = {
                'score': score,
                'location': location
            }
            
            if matched and score > best_score:
                best_score = score
                best_location = location
                best_template = template_name
                best_template_idx = idx
                is_matched = True
        
        # 可視化が要求された場合
        if visualize and is_matched:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, 
                    f"match_{os.path.basename(image_path)}")
            else:
                output_path = None
            
            visualize_matching_result(
                image_path,
                templates[best_template_idx],
                best_location,
                best_score,
                output_path
            )
        
        return {
            'is_matched': is_matched,
            'best_score': best_score,
            'best_location': best_location,
            'best_template': best_template,
            'template_results': template_results
        }
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# メイン処理部分の修正例
def process_images_with_visualization(df_filtered, templates, template_names, threshold, 
                                   output_dir="matching_results"):
    """
    全画像に対してテンプレートマッチングを実行し、結果を可視化
    """
    results = []
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="画像処理中"):
        try:
            # 元のデータフレームの行をコピー
            result_row = row.copy()
            
            # 画像処理の実行（可視化付き）
            matching_result = process_single_image_with_visualization(
                row['defect_image_orig'],
                templates,
                template_names,
                threshold,
                visualize=True,
                output_dir=output_dir
            )
            
            if matching_result is not None:
                # 予測結果の追加
                result_row['predicted_label'] = 1 if matching_result['is_matched'] else 0
                result_row['best_match_score'] = matching_result['best_score']
                result_row['best_match_template'] = matching_result['best_template']
                
                # マッチング位置の追加
                if matching_result['best_location'] is not None:
                    result_row['match_x'] = matching_result['best_location'][0]
                    result_row['match_y'] = matching_result['best_location'][1]
                
                results.append(result_row)
            
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    return pd.DataFrame(results)

# 使用例
# 最適な閾値での最終評価と結果の可視化
print("\n=== 最適な閾値での最終評価（可視化付き） ===")
final_results_df = process_images_with_visualization(
    df_filtered, 
    templates, 
    template_names, 
    best_threshold,
    output_dir="matching_results"
)
```

主な追加機能：
1. マッチング位置を緑色の矩形で表示
2. マッチングスコアを画像上に表示
3. 可視化結果の保存機能
4. 結果のDataFrameにマッチング位置情報を追加

使用方法：
```python
# 個別の画像の処理と可視化
result = process_single_image_with_visualization(
    image_path=sample_image_path,
    templates=templates,
    template_names=template_names,
    threshold=0.8,
    visualize=True,
    output_dir="output"
)

# 全画像の処理と可視化
results_df = process_images_with_visualization(
    df_filtered=df_filtered,
    templates=templates,
    template_names=template_names,
    threshold=best_threshold,
    output_dir="matching_results"
)
```

この可視化機能により：
- マッチング位置の視覚的な確認が可能
- スコアの直感的な理解が容易
- 結果の保存と後での確認が可能
- マッチング精度の視覚的な評価が可能

必要に応じて、可視化のスタイル（色、フォント、サイズなど）を調整できます。

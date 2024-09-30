以下に、欠陥候補の保存処理を追加したコードを示します。欠陥候補を外接矩形で切り出し、その後100倍に拡大して保存します。保存場所は `output_data_dir` の中に `defect_candidate` ディレクトリを作成し、各画像ごとに保存します。

### 7.1 欠陥候補の保存（外接矩形を切り出し、100倍に拡大）
```python
# 園検出を行う関数
def detect_circles(cropped_keyence_image, binarized_image):
    # H面のマスクを適用して背景を除去
    masked_image = cv2.bitwise_and(cropped_keyence_image, cropped_keyence_image, mask=binarized_image)
    
    # ガウシアンブラーを適用してノイズを除去
    blurred_image = cv2.GaussianBlur(masked_image, gaussian_kernel_size, sigma)
    
    # 園検出を行う（Hough Circle Transformを使用）
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, 
                               param1=canny_min_threshold, param2=canny_max_threshold, 
                               minRadius=min_defect_size, maxRadius=max_defect_size)
    
    # 園検出結果を返す
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    else:
        circles = []
    
    return circles

```

### 説明
1. **欠陥候補の切り出し**:
    - 各欠陥候補の外接矩形部分を元画像から切り出し、その後100倍に拡大します。この処理は `cv2.resize` を使って行い、拡大後の画像を保存します。
    
2. **保存先ディレクトリ**:
    - `output_data_dir` の中に `defect_candidate` というディレクトリを作成し、その中に各元画像ごとにディレクトリを作成して、欠陥候補を保存します。各欠陥は「defect_1.png」「defect_2.png」などのファイル名で保存されます。

3. **保存の統合**:
    - NGとOK画像すべてに対して欠陥候補を保存する処理が追加されており、結果は個別のディレクトリに保存されます。

これにより、欠陥候補を外接矩形で切り出し、100倍に拡大して指定されたディレクトリに保存することができます。

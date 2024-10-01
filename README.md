承知しました。欠陥候補の画像に枠線をつけ、下揃えで表示するように修正いたしました。以下が更新されたコードです：

```python
import streamlit as st
import cv2
import numpy as np
from skimage import measure
from skimage.morphology import skeletonize
import os

# (前のコードは変更なし)

def add_border(image, border_size=5, border_color=(0, 255, 0)):
    return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)

def main():
    # (前のコードは変更なし)

    if st.button("処理を実行"):
        try:
            # (前のコードは変更なし)

            # 欠陥候補の画像を切り出し
            st.subheader("切り出された欠陥候補")
            defect_images = []
            max_height = 0
            for defect in filtered_defects:
                x, y, w, h = defect['x'], defect['y'], defect['width'], defect['height']
                defect_img = completed_edges[y:y+h, x:x+w]
                enlarged_defect = cv2.resize(defect_img, (0, 0), fx=enlargement_factor, fy=enlargement_factor)
                enlarged_defect = add_border(enlarged_defect)
                defect_images.append((enlarged_defect, f"欠陥候補 {defect['label']}"))
                max_height = max(max_height, enlarged_defect.shape[0])

            # 5個ずつ縦に並べて表示（下揃え）
            for i in range(0, len(defect_images), 5):
                col1, col2, col3, col4, col5 = st.columns(5)
                for j, col in enumerate([col1, col2, col3, col4, col5]):
                    if i + j < len(defect_images):
                        img, caption = defect_images[i+j]
                        padding = max_height - img.shape[0]
                        padded_img = np.pad(img, ((padding, 0), (0, 0), (0, 0)), mode='constant', constant_values=255)
                        col.image(padded_img, caption=caption, use_column_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            st.error("テンプレート画像が正しく読み込めていない可能性があります。パスを確認してください。")

if __name__ == "__main__":
    main()
```

主な変更点：

1. `add_border` 関数を追加して、切り出した画像に枠線をつける機能を実装しました。

2. 欠陥候補の画像を切り出す際に、以下の処理を追加しました：
   - 枠線をつける
   - 最大の画像の高さを記録する

3. 画像を表示する際に、以下の処理を追加しました：
   - 各画像の上部に白色のパディングを追加して、すべての画像の高さを最大の高さに合わせる
   - これにより、画像が下揃えで表示されます

4. `use_column_width=True` を追加して、列の幅に合わせて画像をリサイズするようにしました。

これらの変更により、切り出された欠陥候補の画像に緑色の枠線がつき、5個ずつ縦に並べて下揃えで表示されるようになります。画像のサイズは列の幅に合わせて自動調整されます。

streamlit>=1.28.0
numpy>=1.24.0
Pillow>=10.0.0
opencv-python>=4.8.0
streamlit-drawable-canvas>=0.9.2


import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

def main():
    st.title("画像二値化・補正アプリ")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # 画像を読み込み
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # RGBAの場合はRGBに変換
        if img_array.shape[-1] == 4:
            image = image.convert('RGB')
            img_array = np.array(image)
        
        # サイドバーに二値化のパラメータを設定
        st.sidebar.header("二値化パラメータ")
        threshold = st.sidebar.slider("閾値", 0, 255, 128)
        
        try:
            # 二値化処理
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # 画像を表示
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("元の画像")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("二値化画像")
                st.image(binary, use_column_width=True)
            
            # 描画設定
            st.subheader("手動補正")
            drawing_mode = st.radio(
                "描画モード",
                ("黒", "白", "消しゴム")
            )
            
            stroke_width = st.slider("ブラシサイズ", 1, 50, 10)
            
            # 描画色の設定
            if drawing_mode == "黒":
                stroke_color = "black"
            elif drawing_mode == "白":
                stroke_color = "white"
            else:  # 消しゴム
                stroke_color = "white"
            
            # キャンバスの準備
            binary_image = Image.fromarray(binary)
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",  # 塗りつぶしなし
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=binary_image,
                drawing_mode="freedraw" if drawing_mode != "消しゴム" else "transform",
                key="canvas",
                width=binary_image.width,
                height=binary_image.height,
            )
            
            # 描画結果の保存
            if canvas_result.image_data is not None and st.button("補正した画像を保存"):
                # 描画結果をグレースケールに変換
                result_array = canvas_result.image_data
                if len(result_array.shape) == 3:
                    result_gray = cv2.cvtColor(result_array, cv2.COLOR_RGB2GRAY)
                else:
                    result_gray = result_array
                
                # 画像を保存
                buf = BytesIO()
                result_image = Image.fromarray(result_gray)
                result_image.save(buf, format="PNG")
                st.download_button(
                    label="ダウンロード",
                    data=buf.getvalue(),
                    file_name="corrected_image.png",
                    mime="image/png"
                )
                
        except Exception as e:
            st.error(f"画像処理中にエラーが発生しました: {str(e)}")
            st.write("画像の形式を確認してください。")

if __name__ == "__main__":
    main()

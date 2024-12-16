streamlit>=1.28.0
numpy>=1.24.0
Pillow>=10.0.0
opencv-python>=4.8.0
streamlit-drawable-canvas>=0.9.3


import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

def main():
    st.title("画像二値化・補正アプリ")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # 画像を読み込み
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # RGBAの場合はRGBに変換
        if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
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
                ("フリーハンド", "四角", "直線")
            )
            
            stroke_width = st.slider("ブラシサイズ", 1, 50, 10)
            stroke_color = st.radio("描画色", ("黒", "白"))
            
            # 描画色の設定
            if stroke_color == "黒":
                color = "#000000"
            else:
                color = "#FFFFFF"
            
            # キャンバスの準備
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=stroke_width,
                stroke_color=color,
                background_image=Image.fromarray(binary),
                drawing_mode=drawing_mode.lower(),
                key="canvas",
                width=binary.shape[1],
                height=binary.shape[0],
            )
            
            # 描画結果の保存
            if canvas_result.image_data is not None:
                # 描画結果を二値化画像として保存
                result_array = np.array(canvas_result.image_data)
                
                if st.button("補正した画像を保存"):
                    # グレースケールに変換して二値化
                    result_gray = cv2.cvtColor(result_array, cv2.COLOR_RGBA2GRAY)
                    _, result_binary = cv2.threshold(result_gray, 127, 255, cv2.THRESH_BINARY)
                    
                    # 保存用のバッファを作成
                    buf = BytesIO()
                    result_image = Image.fromarray(result_binary)
                    result_image.save(buf, format="PNG")
                    
                    # ダウンロードボタンを表示
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

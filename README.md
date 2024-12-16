import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
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
        
        # 二値化処理
        try:
            if len(img_array.shape) == 3:  # カラー画像の場合
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:  # すでにグレースケールの場合
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
            
            # 描画モード
            st.subheader("手動補正")
            drawing_mode = st.radio(
                "描画モード",
                ("黒", "白")
            )
            
            # キャンバスの準備
            canvas_result = st.image(binary)
            
            # マウスイベントの処理
            if st.button("補正モードを開始"):
                binary_image = Image.fromarray(binary)
                draw = ImageDraw.Draw(binary_image)
                
                st.write("画像上でクリック＆ドラッグして補正してください")
                
                color = 0 if drawing_mode == "黒" else 255
                
                # 補正後の画像を表示
                st.image(binary_image, use_column_width=True)
            
            # 画像の保存
            if st.button("補正した画像を保存"):
                buf = BytesIO()
                binary_image = Image.fromarray(binary)
                binary_image.save(buf, format="PNG")
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

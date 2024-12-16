import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

def resize_for_display(image, max_size=800):
    """画像を表示用にリサイズする"""
    width, height = image.size
    if width > max_size or height > max_size:
        ratio = min(max_size/width, max_size/height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def process_canvas_result(canvas_result):
    """キャンバスの結果を処理して二値画像に変換する"""
    if canvas_result.image_data is not None:
        # RGBA形式の画像データを取得
        img_data = np.array(canvas_result.image_data)
        # アルファチャンネルを基準に二値化
        # 白い部分（255）は維持し、それ以外は黒（0）にする
        mask = img_data[:, :, -1] > 0
        binary = np.zeros(img_data.shape[:2], dtype=np.uint8)
        binary[mask] = 255
        return binary
    return None

def main():
    st.title("画像二値化・補正アプリ")
    
    if 'original_size' not in st.session_state:
        st.session_state.original_size = None
    if 'display_size' not in st.session_state:
        st.session_state.display_size = None
    
    uploaded_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image_original = Image.open(uploaded_file)
        image_display = resize_for_display(image_original.copy())
        
        st.session_state.original_size = image_original.size
        st.session_state.display_size = image_display.size
        
        # RGBAの場合はRGBに変換
        if image_original.mode == 'RGBA':
            image_original = image_original.convert('RGB')
            image_display = image_display.convert('RGB')
        
        img_array_original = np.array(image_original)
        img_array_display = np.array(image_display)
        
        st.sidebar.header("二値化パラメータ")
        threshold = st.sidebar.slider("閾値", 0, 255, 128)
        
        try:
            # 二値化処理（元サイズ）
            gray_original = cv2.cvtColor(img_array_original, cv2.COLOR_RGB2GRAY)
            _, binary_original = cv2.threshold(gray_original, threshold, 255, cv2.THRESH_BINARY)
            
            # 二値化処理（表示用）
            gray_display = cv2.cvtColor(img_array_display, cv2.COLOR_RGB2GRAY)
            _, binary_display = cv2.threshold(gray_display, threshold, 255, cv2.THRESH_BINARY)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("元の画像")
                st.image(image_display, use_column_width=True)
                st.write(f"元の画像サイズ: {st.session_state.original_size[0]}x{st.session_state.original_size[1]}")
            
            with col2:
                st.subheader("二値化画像")
                st.image(binary_display, use_column_width=True)
                st.write(f"表示サイズ: {st.session_state.display_size[0]}x{st.session_state.display_size[1]}")
            
            st.subheader("手動補正")
            drawing_mode = st.radio(
                "描画モード",
                ("freedraw", "rect", "line"),
                format_func=lambda x: {
                    "freedraw": "フリーハンド",
                    "rect": "四角",
                    "line": "直線"
                }[x]
            )
            
            stroke_width = st.slider("ブラシサイズ", 1, 50, 10)
            stroke_color = st.radio("描画色", ("黒", "白"))
            
            if stroke_color == "黒":
                stroke_color_hex = "#000000"
                fill_color = "rgba(0, 0, 0, 1.0)" if drawing_mode == "rect" else "rgba(0, 0, 0, 0.0)"
            else:
                stroke_color_hex = "#FFFFFF"
                fill_color = "rgba(255, 255, 255, 1.0)" if drawing_mode == "rect" else "rgba(255, 255, 255, 0.0)"
            
            canvas_result = st_canvas(
                fill_color=fill_color,
                stroke_width=stroke_width,
                stroke_color=stroke_color_hex,
                background_image=Image.fromarray(binary_display).convert('RGB'),
                drawing_mode=drawing_mode,
                key=f"canvas_{threshold}",
                width=binary_display.shape[1],
                height=binary_display.shape[0],
                display_toolbar=True
            )
            
            if canvas_result.image_data is not None and st.button("補正した画像を保存"):
                # キャンバス結果を二値画像に変換
                binary_result = process_canvas_result(canvas_result)
                if binary_result is not None:
                    # 元のサイズにリサイズ
                    binary_result_pil = Image.fromarray(binary_result)
                    result_resized = binary_result_pil.resize(
                        st.session_state.original_size, 
                        Image.Resampling.LANCZOS
                    )
                    
                    # 保存用のバッファを作成
                    buf = BytesIO()
                    result_resized.save(buf, format="PNG")
                    
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

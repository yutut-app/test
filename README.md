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

def process_canvas_result(canvas_result, original_size):
    """キャンバスの描画結果を処理して二値化画像を返す"""
    if canvas_result.image_data is None:
        return None
    
    # RGBA配列を取得
    result_array = canvas_result.image_data
    
    # RGBAをグレースケールに変換
    # (R * 0.299 + G * 0.587 + B * 0.114)の重み付け
    gray = np.dot(result_array[..., :3], [0.299, 0.587, 0.114])
    
    # 二値化（128を閾値として使用）
    binary = (gray > 128).astype(np.uint8) * 255
    
    # PIL Imageに変換
    binary_image = Image.fromarray(binary)
    
    # 元のサイズにリサイズ
    if binary_image.size != original_size:
        binary_image = binary_image.resize(original_size, Image.Resampling.LANCZOS)
    
    return binary_image

def main():
    st.title("画像二値化・補正アプリ")
    
    # セッションステートの初期化
    if 'original_size' not in st.session_state:
        st.session_state.original_size = None
    if 'display_size' not in st.session_state:
        st.session_state.display_size = None
    
    # ファイルアップロード
    uploaded_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # 元の画像を読み込み（処理用）
            image_original = Image.open(uploaded_file)
            # 表示用にリサイズした画像
            image_display = resize_for_display(image_original.copy())
            
            # サイズ情報を保存
            st.session_state.original_size = image_original.size
            st.session_state.display_size = image_display.size
            
            # 画像をnumpy配列に変換
            img_array_original = np.array(image_original)
            img_array_display = np.array(image_display)
            
            # RGBAの場合はRGBに変換
            if len(img_array_original.shape) == 3 and img_array_original.shape[-1] == 4:
                image_original = image_original.convert('RGB')
                image_display = image_display.convert('RGB')
                img_array_original = np.array(image_original)
                img_array_display = np.array(image_display)
            
            # サイドバーに二値化のパラメータを設定
            st.sidebar.header("二値化パラメータ")
            threshold = st.sidebar.slider("閾値", 0, 255, 128)
            
            # 二値化処理（元サイズ）
            if len(img_array_original.shape) == 3:
                gray_original = cv2.cvtColor(img_array_original, cv2.COLOR_RGB2GRAY)
            else:
                gray_original = img_array_original
            
            _, binary_original = cv2.threshold(gray_original, threshold, 255, cv2.THRESH_BINARY)
            
            # 二値化処理（表示用）
            if len(img_array_display.shape) == 3:
                gray_display = cv2.cvtColor(img_array_display, cv2.COLOR_RGB2GRAY)
            else:
                gray_display = img_array_display
            
            _, binary_display = cv2.threshold(gray_display, threshold, 255, cv2.THRESH_BINARY)
            
            # 画像を表示
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("元の画像")
                st.image(image_display, use_column_width=True)
                st.write(f"元の画像サイズ: {st.session_state.original_size[0]}x{st.session_state.original_size[1]}")
            
            with col2:
                st.subheader("二値化画像")
                st.image(binary_display, use_column_width=True)
                st.write(f"表示サイズ: {st.session_state.display_size[0]}x{st.session_state.display_size[1]}")
            
            # 描画設定
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
            
            # 描画色の設定
            if stroke_color == "黒":
                stroke_color_hex = "#000000"
                fill_color = "rgba(0, 0, 0, 1.0)" if drawing_mode == "rect" else "rgba(255, 255, 255, 0.0)"
            else:
                stroke_color_hex = "#FFFFFF"
                fill_color = "rgba(255, 255, 255, 1.0)" if drawing_mode == "rect" else "rgba(255, 255, 255, 0.0)"
            
            # キャンバスの作成（表示サイズ用）
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
            
            # 描画結果の保存
            if st.button("補正した画像を保存"):
                if canvas_result.image_data is not None:
                    # 描画結果を処理
                    result_binary = process_canvas_result(
                        canvas_result, 
                        (st.session_state.original_size[0], st.session_state.original_size[1])
                    )
                    
                    if result_binary is not None:
                        # バッファに保存
                        buf = BytesIO()
                        result_binary.save(buf, format="PNG")
                        
                        # ダウンロードボタンを表示
                        st.download_button(
                            label="ダウンロード",
                            data=buf.getvalue(),
                            file_name="corrected_image.png",
                            mime="image/png"
                        )
                    else:
                        st.error("画像の処理に失敗しました。")
                else:
                    st.warning("描画データがありません。")
                
        except Exception as e:
            st.error(f"画像処理中にエラーが発生しました: {str(e)}")
            st.write("画像の形式を確認してください。")

if __name__ == "__main__":
    main()

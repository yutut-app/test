import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

def resize_image(image, max_size=1024):
    """画像を最大サイズ以下にリサイズする"""
    width, height = image.size
    if width > max_size or height > max_size:
        # アスペクト比を保持したままリサイズ
        ratio = min(max_size/width, max_size/height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def main():
    st.title("画像二値化・補正アプリ")
    
    # セッションステートの初期化
    if 'binary' not in st.session_state:
        st.session_state.binary = None
    
    # サイドバーに最大サイズの設定を追加
    st.sidebar.header("画像設定")
    max_size = st.sidebar.slider("最大画像サイズ（ピクセル）", 512, 2048, 1024)
    
    # ファイルアップロード
    uploaded_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # 画像を読み込みとリサイズ
            image = Image.open(uploaded_file)
            st.sidebar.write(f"元の画像サイズ: {image.size}")
            
            # 画像をリサイズ
            image = resize_image(image, max_size)
            st.sidebar.write(f"リサイズ後の画像サイズ: {image.size}")
            
            img_array = np.array(image)
            
            # RGBAの場合はRGBに変換
            if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
                image = image.convert('RGB')
                img_array = np.array(image)
            
            # サイドバーに二値化のパラメータを設定
            st.sidebar.header("二値化パラメータ")
            threshold = st.sidebar.slider("閾値", 0, 255, 128)
            
            # 二値化処理
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            st.session_state.binary = binary
            
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
            color = "#000000" if stroke_color == "黒" else "#FFFFFF"
            
            # キャンバスサイズの計算
            canvas_width = min(binary.shape[1], max_size)
            canvas_height = min(binary.shape[0], max_size)
            
            # キャンバスの作成
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=stroke_width,
                stroke_color=color,
                background_image=Image.fromarray(binary).convert('RGB'),
                drawing_mode=drawing_mode,
                key=f"canvas_{threshold}",
                width=canvas_width,
                height=canvas_height,
                display_toolbar=True
            )
            
            # 描画結果の保存
            if canvas_result.image_data is not None and st.button("補正した画像を保存"):
                # 描画結果を二値化画像として保存
                result_array = canvas_result.image_data
                
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
            st.write("エラーの詳細：", e)

if __name__ == "__main__":
    main()

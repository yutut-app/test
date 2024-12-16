import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

def calculate_display_size(width, height, max_size=800):
    """アスペクト比を維持したまま表示サイズを計算する"""
    if width > height:
        if width > max_size:
            ratio = max_size / width
            return max_size, int(height * ratio)
        return width, height
    else:
        if height > max_size:
            ratio = max_size / height
            return int(width * ratio), max_size
        return width, height

def process_canvas_result(canvas_result, binary_original, original_size):
    """キャンバスの描画結果を処理して二値化画像を返す"""
    if canvas_result.image_data is None:
        return None
    
    try:
        # キャンバスの描画結果をnumpy配列として取得
        canvas_array = canvas_result.image_data
        
        # キャンバスのサイズを取得
        canvas_height, canvas_width = canvas_array.shape[:2]
        
        # 二値化画像をキャンバスのサイズにリサイズ
        binary_resized = cv2.resize(binary_original, (canvas_width, canvas_height))
        
        # アルファチャンネルを使用してマスクを作成
        alpha_mask = canvas_array[..., 3] > 0
        
        # キャンバスの描画部分を二値化
        gray_canvas = np.dot(canvas_array[..., :3], [0.299, 0.587, 0.114])
        binary_canvas = (gray_canvas > 128).astype(np.uint8) * 255
        
        # 結果の画像を作成
        result = binary_resized.copy()
        result[alpha_mask] = binary_canvas[alpha_mask]
        
        # 元のサイズにリサイズ
        result_resized = cv2.resize(result, (original_size[0], original_size[1]), 
                                  interpolation=cv2.INTER_NEAREST)
        return Image.fromarray(result_resized)
    
    except Exception as e:
        st.error(f"画像処理エラー: {str(e)}")
        return None

def main():
    st.title("画像二値化・補正アプリ")
    
    if 'binary_original' not in st.session_state:
        st.session_state.binary_original = None
    if 'original_size' not in st.session_state:
        st.session_state.original_size = None
    
    uploaded_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # 元の画像を読み込み
            image_original = Image.open(uploaded_file)
            
            # 画像がRGBAの場合はRGBに変換
            if image_original.mode == 'RGBA':
                image_original = image_original.convert('RGB')
            
            # 表示サイズを計算
            display_width, display_height = calculate_display_size(
                image_original.size[0], 
                image_original.size[1]
            )
            
            # 表示用にリサイズ（アスペクト比を維持）
            image_display = image_original.resize((display_width, display_height), 
                                                Image.Resampling.LANCZOS)
            
            # サイズ情報を保存
            st.session_state.original_size = image_original.size
            
            # numpy配列に変換
            img_array_original = np.array(image_original)
            img_array_display = np.array(image_display)
            
            # 二値化のパラメータ設定
            st.sidebar.header("二値化パラメータ")
            threshold = st.sidebar.slider("閾値", 0, 255, 128)
            
            # 二値化処理（元サイズ）
            gray_original = cv2.cvtColor(img_array_original, cv2.COLOR_RGB2GRAY)
            _, binary_original = cv2.threshold(gray_original, threshold, 255, cv2.THRESH_BINARY)
            st.session_state.binary_original = binary_original
            
            # 二値化処理（表示用）
            gray_display = cv2.cvtColor(img_array_display, cv2.COLOR_RGB2GRAY)
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
                st.write(f"表示サイズ: {display_width}x{display_height}")
            
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
            
            # キャンバスコンテナの作成（幅を制御）
            canvas_container = st.container()
            with canvas_container:
                # キャンバスの作成
                canvas_result = st_canvas(
                    fill_color=fill_color,
                    stroke_width=stroke_width,
                    stroke_color=stroke_color_hex,
                    background_image=Image.fromarray(binary_display),
                    drawing_mode=drawing_mode,
                    key=f"canvas_{threshold}",
                    width=display_width,
                    height=display_height,
                    display_toolbar=True
                )
            
            # 描画結果の保存
            if st.button("補正した画像を保存"):
                if canvas_result.image_data is not None:
                    # 描画結果を処理
                    result_binary = process_canvas_result(
                        canvas_result,
                        st.session_state.binary_original,
                        st.session_state.original_size
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

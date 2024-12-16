import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

def calculate_display_size(width, height, max_width=700, max_height=500):
    """アスペクト比を維持しながら表示サイズを計算する"""
    aspect_ratio = width / height
    
    # 最大幅に基づいてリサイズ
    new_width = min(width, max_width)
    new_height = int(new_width / aspect_ratio)
    
    # 高さが最大値を超える場合は高さに基づいてリサイズ
    if new_height > max_height:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)
    
    return new_width, new_height

def process_canvas_result(canvas_result, binary_original, original_size):
    """キャンバスの描画結果を処理して二値化画像を返す"""
    if canvas_result.image_data is None:
        return None
    
    # キャンバスのサイズを取得
    canvas_height, canvas_width = canvas_result.image_data.shape[:2]
    
    # 二値化画像をキャンバスのサイズにリサイズ
    binary_resized = cv2.resize(binary_original, (canvas_width, canvas_height))
    
    # キャンバスの描画結果をnumpy配列として取得
    canvas_array = canvas_result.image_data
    
    # アルファチャンネルを使用してマスクを作成
    alpha_mask = canvas_array[..., 3] > 0
    
    # キャンバスの描画部分を二値化
    gray_canvas = np.dot(canvas_array[..., :3], [0.299, 0.587, 0.114])
    binary_canvas = (gray_canvas > 128).astype(np.uint8) * 255
    
    # 結果の画像を作成
    result = binary_resized.copy()
    result[alpha_mask] = binary_canvas[alpha_mask]
    
    # 元のサイズにリサイズ
    result_resized = cv2.resize(result, (original_size[0], original_size[1]), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(result_resized)

def main():
    st.title("画像二値化・補正アプリ")
    
    # セッションステートの初期化
    if 'binary_original' not in st.session_state:
        st.session_state.binary_original = None
    if 'original_size' not in st.session_state:
        st.session_state.original_size = None
    if 'display_size' not in st.session_state:
        st.session_state.display_size = None
    if 'last_threshold' not in st.session_state:
        st.session_state.last_threshold = 128
    if 'gray_original' not in st.session_state:
        st.session_state.gray_original = None
    
    uploaded_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # 元の画像を読み込み
            image_original = Image.open(uploaded_file)
            
            # 表示サイズを計算
            display_width, display_height = calculate_display_size(
                image_original.size[0], 
                image_original.size[1]
            )
            st.session_state.display_size = (display_width, display_height)
            
            # 表示用にリサイズ（アスペクト比を維持）
            image_display = image_original.resize(
                st.session_state.display_size,
                Image.Resampling.LANCZOS
            )
            
            # サイズ情報を保存
            st.session_state.original_size = image_original.size
            
            # 画像をnumpy配列に変換
            img_array_original = np.array(image_original)
            img_array_display = np.array(image_display)
            
            # RGBAの場合はRGBに変換
            if len(img_array_original.shape) == 3 and img_array_original.shape[-1] == 4:
                image_original = image_original.convert('RGB')
                image_display = image_display.convert('RGB')
                img_array_original = np.array(image_original)
                img_array_display = np.array(image_display)
            
            # グレースケール変換（元サイズ）
            if len(img_array_original.shape) == 3:
                if st.session_state.gray_original is None:
                    st.session_state.gray_original = cv2.cvtColor(img_array_original, cv2.COLOR_RGB2GRAY)
            else:
                if st.session_state.gray_original is None:
                    st.session_state.gray_original = img_array_original
            
            # 二値化のパラメータ設定
            st.sidebar.header("二値化パラメータ")
            threshold = st.sidebar.slider("閾値", 0, 255, 128)
            
            # 閾値が変更された場合、新しい二値化画像を生成
            if threshold != st.session_state.last_threshold:
                _, binary_original = cv2.threshold(st.session_state.gray_original, threshold, 255, cv2.THRESH_BINARY)
                st.session_state.binary_original = binary_original
                st.session_state.last_threshold = threshold
            
            # 表示用の二値化処理
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
            
            # キャンバスの作成（アスペクト比を維持）
            canvas_result = st_canvas(
                fill_color=fill_color,
                stroke_width=stroke_width,
                stroke_color=stroke_color_hex,
                background_image=Image.fromarray(binary_display),
                drawing_mode=drawing_mode,
                key=f"canvas_{threshold}",  # 閾値が変わるたびにキャンバスをリセット
                width=display_width,
                height=display_height,
                display_toolbar=True
            )
            
            # 描画結果の保存
            if st.button("補正した画像を保存"):
                if canvas_result.image_data is not None:
                    try:
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
                    except Exception as e:
                        st.error(f"画像の保存中にエラーが発生しました: {str(e)}")
                else:
                    st.warning("描画データがありません。")
                
        except Exception as e:
            st.error(f"画像処理中にエラーが発生しました: {str(e)}")
            st.write("画像の形式を確認してください。")

if __name__ == "__main__":
    main()

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

def create_binary_image(image, threshold):
    """画像を二値化する"""
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

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
    if 'image_original' not in st.session_state:
        st.session_state.image_original = None
    if 'original_size' not in st.session_state:
        st.session_state.original_size = None
    if 'display_size' not in st.session_state:
        st.session_state.display_size = None
    if 'last_threshold' not in st.session_state:
        st.session_state.last_threshold = None
    if 'current_binary_original' not in st.session_state:
        st.session_state.current_binary_original = None
    
    # ステップ1: 画像のアップロード
    uploaded_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # 画像が新しくアップロードされた場合
            if st.session_state.image_original is None:
                image_original = Image.open(uploaded_file)
                
                # RGBAの場合はRGBに変換
                if image_original.mode == 'RGBA':
                    image_original = image_original.convert('RGB')
                
                st.session_state.image_original = image_original
                
                # 表示サイズを計算
                display_width, display_height = calculate_display_size(
                    image_original.size[0], 
                    image_original.size[1]
                )
                st.session_state.display_size = (display_width, display_height)
                st.session_state.original_size = image_original.size
            
            # ステップ2: 二値化パラメータの設定
            st.sidebar.header("二値化パラメータ")
            threshold = st.sidebar.slider("閾値", 0, 255, 128)
            
            # 閾値が変更された場合の処理
            if st.session_state.last_threshold != threshold:
                st.session_state.last_threshold = threshold
                # 新しい閾値で二値化画像を生成
                binary_original = create_binary_image(st.session_state.image_original, threshold)
                st.session_state.current_binary_original = binary_original
                # キャンバスをリセット
                if 'canvas_key' in st.session_state:
                    st.session_state.pop('canvas_key')
            
            # 現在の二値化画像がない場合は生成
            if st.session_state.current_binary_original is None:
                binary_original = create_binary_image(st.session_state.image_original, threshold)
                st.session_state.current_binary_original = binary_original
            
            # 表示用画像の準備
            image_display = st.session_state.image_original.resize(
                st.session_state.display_size,
                Image.Resampling.LANCZOS
            )
            
            # 表示用二値化画像の準備
            binary_display = cv2.resize(
                st.session_state.current_binary_original,
                (st.session_state.display_size[0], st.session_state.display_size[1])
            )
            
            # 画像表示
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("元の画像")
                st.image(image_display, use_column_width=True)
                st.write(f"元の画像サイズ: {st.session_state.original_size[0]}x{st.session_state.original_size[1]}")
            
            with col2:
                st.subheader("二値化画像")
                st.image(binary_display, use_column_width=True)
                st.write(f"表示サイズ: {st.session_state.display_size[0]}x{st.session_state.display_size[1]}")
            
            # ステップ3: 手動補正
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
            
            # キャンバスの作成
            canvas_key = f"canvas_{threshold}_{st.session_state.get('canvas_key', 0)}"
            canvas_result = st_canvas(
                fill_color=fill_color,
                stroke_width=stroke_width,
                stroke_color=stroke_color_hex,
                background_image=Image.fromarray(binary_display),
                drawing_mode=drawing_mode,
                key=canvas_key,
                width=st.session_state.display_size[0],
                height=st.session_state.display_size[1],
                display_toolbar=True
            )
            
            # ステップ4: 画像の保存
            if st.button("補正した画像を保存"):
                if canvas_result.image_data is not None:
                    try:
                        # 描画結果を処理
                        result_binary = process_canvas_result(
                            canvas_result,
                            st.session_state.current_binary_original,  # 現在の閾値で生成された二値化画像を使用
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

streamlit==1.24.0
numpy>=1.24.0
Pillow>=6.2.0,<10.0.0
opencv-python>=4.8.0
streamlit-drawable-canvas==0.8.0

# 既存のパッケージをアンインストール
pip uninstall streamlit streamlit-drawable-canvas Pillow

# requirements.txtからインストール
pip install -r requirements.txt

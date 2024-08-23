# data_loader.py

import pandas as pd
from config import INPUT_FILE_PATH, DATE_FORMAT

def load_casting_data():
    """
    CSVファイルから鋳造工程のデータを読み込む関数

    Returns:
        pandas.DataFrame: 読み込んだデータ
    """
    try:
        data = pd.read_csv(INPUT_FILE_PATH)
        
        # 日時列を日時型に変換
        date_columns = ['日時', '出荷検査日時', '加工検査日時']
        for col in date_columns:
            data[col] = pd.to_datetime(data[col], format=DATE_FORMAT)
        
        return data
    except Exception as e:
        print(f"データの読み込みに失敗しました: {e}")
        return None
        

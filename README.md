# 鋳造条件の列名を取得（int型とfloat型の列）
CASTING_CONDITION_COLUMNS = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
CASTING_CONDITION_COLUMNS = [col for col in CASTING_CONDITION_COLUMNS if col != '目的変数']

# 出力ディレクトリの作成
OUTPUT_DIR = r'..\data\output\eda\NG数の時系列の偏り'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 現在の日時を取得（ファイル名用）
CURRENT_TIME = datetime.now().strftime("%y%m%d%H%M")

# 品番ごとの固定色を定義
PRODUCT_COLOR_MAP = {
    2: 'red',
    4: 'blue',
    5: 'green',
    6: 'purple'
}

# グラフを表示するかどうかのフラグ
SHOW_PLOTS_DAILY = False
SHOW_PLOTS_HOURLY = False
SHOW_PLOTS_WEEKDAY = False
SHOW_PLOTS_WEEKLY = False

def calculate_ng_rate(group):
    """NG率を計算する関数

    Args:
        group (pandas.DataFrame): グループ化されたデータフレーム

    Returns:
        tuple: (NG数, 総数, NG率)のタプル。データがない場合はNone
    """
    total = len(group)
    if total == 0:
        return None
    ng_count = (group['目的変数'] == 1).sum()
    ng_rate = (ng_count / total) * 100 if total > 0 else 0
    return ng_count, total, ng_rate

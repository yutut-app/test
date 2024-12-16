def load_template_images():
    """
    左右判定用とマスク用のテンプレート画像を読み込みます
    
    戻り値:
        tuple: (左右判定テンプレート画像, マスクテンプレート画像)のタプル
        各要素は{left: image, right: image}の辞書
    """
    # パスの存在確認
    paths = {
        'left_judge': left_judge_template_path,
        'right_judge': right_judge_template_path,
        'left_mask': left_mask_template_path,
        'right_mask': right_mask_template_path
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"テンプレート画像が見つかりません: {path}")
            return None, None
    
    # 左右判定用テンプレート
    judge_templates = {
        'left': cv2.imread(left_judge_template_path, cv2.IMREAD_GRAYSCALE),
        'right': cv2.imread(right_judge_template_path, cv2.IMREAD_GRAYSCALE)
    }
    
    # マスク用テンプレート
    mask_templates = {
        'left': cv2.imread(left_mask_template_path, cv2.IMREAD_GRAYSCALE),
        'right': cv2.imread(right_mask_template_path, cv2.IMREAD_GRAYSCALE)
    }
    
    # 読み込み確認
    if any(img is None for img in judge_templates.values()) or \
       any(img is None for img in mask_templates.values()):
        print("テンプレート画像の読み込みに失敗しました")
        print("以下のパスを確認してください:")
        print(f"left_judge_template_path: {left_judge_template_path}")
        print(f"right_judge_template_path: {right_judge_template_path}")
        print(f"left_mask_template_path: {left_mask_template_path}")
        print(f"right_mask_template_path: {right_mask_template_path}")
        return None, None
    
    return judge_templates, mask_templates

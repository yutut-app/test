# 'image_name' から 'work_id' を生成する関数
def create_work_id(image_name):
    return image_name[:15]

# 'work_id' 列を追加
df['work_id'] = df['image_name'].apply(create_work_id)

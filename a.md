# 各鋳造条件に対してcatplotを作成し、PDFとして保存
for condition in casting_condition_columns:
    plt.figure(figsize=(12, 8))
    
    # OKデータ（目的変数0）のプロット
    sns.stripplot(data=df[df['目的変数'] == 0], x=condition, y='鋳造機名', 
                  color='blue', alpha=0.3, size=3, jitter=0.3)
    
    # NGデータ（目的変数1）のプロット
    sns.stripplot(data=df[df['目的変数'] == 1], x=condition, y='鋳造機名', 
                  color='orange', alpha=1, size=7, jitter=0.3)
    
    plt.title(f'鋳造機名と{condition}の関係')
    plt.legend(['OK', 'NG'])
    plt.tight_layout()
    
    # 現在の日時を取得してファイル名に使用
    current_time = datetime.now().strftime("%y%m%d%H%M")
    filename = f'timevis_{condition}_{current_time}.pdf'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    # plt.show()  # 可視化する際はこのコメントを外してください

print("全ての図表がPDFとして保存されました。")

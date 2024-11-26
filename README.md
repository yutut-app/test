この問題は、OpenCVの依存ライブラリが不足していることが原因です。libGL.so.1が見つからないというエラーは、Ubuntu/Debianベースのシステムでよく発生します。

以下のコマンドで必要なパッケージをインストールすることで解決できます：

Ubuntu/Debian の場合:
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
```

CentOS/RHEL の場合:
```bash
sudo yum install mesa-libGL
```

Amazon Linux/AWS環境の場合:
```bash
sudo yum install mesa-libGL
```

もし上記で解決しない場合は、以下の追加パッケージもインストールしてみてください：
```bash
sudo apt-get install libglib2.0-0
sudo apt-get install libsm6
sudo apt-get install libxext6
sudo apt-get install libxrender1
```

インストール後、Pythonを再起動してから再度`import cv2`を試してみてください。これで問題が解決するはずです。

ご使用の環境（OS）を教えていただけますか？より具体的なアドバイスができるかもしれません。

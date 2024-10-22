このエラーは、OpenCVが依存するライブラリ `libGL.so.1` がインストールされていない、または正しくリンクされていない場合に発生する。

この問題を解決するための手順を示す。

### 1. `libGL.so.1` を提供するパッケージのインストール
`libGL.so.1` は、OpenGLのライブラリで、多くの場合 `mesa` パッケージが提供している。次のコマンドで `mesa` パッケージをインストールする。

#### Ubuntu/Debianの場合
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
```

### 2. コンテナ内でインストールする場合
もしDockerコンテナ内でJupyter Notebookを実行している場合、同様にコンテナ内で `libgl1-mesa-glx` をインストールする必要がある。コンテナ内で次のコマンドを実行する。

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
```

### 3. 確認
インストールが完了したら、再度 `import cv2` を実行してみる。これでエラーが解消されるはずだ。

もし問題が続く場合は、詳細なエラーメッセージや状況を教えてもらえれば、さらに具体的なサポートが可能だ。

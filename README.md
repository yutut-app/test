docker saveの手順

1. 事前準備 (ローカルPC、WSL2のUbuntu):
   a. Windowsのスタートメニューから「Ubuntu」を起動します。
   b. 以下のコマンドを実行して、保存先ディレクトリを作成します：
      ```
      mkdir -p ~/docker_images
      ```

2. Dockerイメージの保存 (ローカルPC、WSL2のUbuntu):
   a. 以下のコマンドを実行して、pullしたJupyter SciPy Notebookイメージをtarファイルとして保存します：
      ```
      docker save jupyter/scipy-notebook:latest > ~/docker_images/scipy-notebook.tar
      ```
   b. このコマンドは、イメージを圧縮せずにtarファイルとして保存します。処理には時間がかかる場合があります。

3. 保存の確認 (ローカルPC、WSL2のUbuntu):
   a. 以下のコマンドを実行して、保存したファイルの存在を確認します：
      ```
      ls -lh ~/docker_images/
      ```
   b. 出力に「scipy-notebook.tar」ファイルが表示され、そのサイズが確認できます。

4. Windowsファイルシステムからのアクセス (ローカルPC、エクスプローラー):
   a. エクスプローラーを開きます。
   b. アドレスバーに以下を入力してEnterを押します：
      ```
      \\wsl$\Ubuntu\home\YourUsername\docker_images
      ```
      注: 'YourUsername'を実際のWSL2 Ubuntuのユーザー名に置き換えてください。
   c. ここで「scipy-notebook.tar」ファイルが見えるはずです。

注意事項:
- docker saveで作成されるtarファイルは通常非常に大きくなります（数GBの可能性があります）。
- 保存には十分なディスク容量があることを確認してください。
- 大きなファイルの転送には時間がかかる場合があります。

これらの手順により、pullしたDockerイメージがtarファイルとして保存され、Windowsファイルシステムからもアクセス可能になります。


1. Dockerのインストール (ローカルPC、WSL2のUbuntu):
   a. Windowsのスタートメニューから「Ubuntu」を起動します。
   b. 以下のコマンドを実行してDockerをインストールします：
      ```
      sudo snap install docker
      ```
   c. インストールが完了するまで待ちます。

2. Dockerグループへのユーザー追加 (ローカルPC、WSL2のUbuntu):
   a. 以下のコマンドを実行して、現在のユーザーをdockerグループに追加します：
      ```
      sudo addgroup --system docker
      sudo adduser $USER docker
      newgrp docker
      ```
   b. 変更を反映するために、Ubuntu端末を一度閉じて再度開きます。

3. Dockerサービスの起動確認 (ローカルPC、WSL2のUbuntu):
   a. 以下のコマンドを実行してDockerサービスの状態を確認します：
      ```
      sudo snap services docker
      ```
   b. サービスが動作していない場合は、以下のコマンドで起動します：
      ```
      sudo snap start docker
      ```

4. wsl-vpnkitの起動 (ローカルPC):
   a. VPNに接続していることを確認します。
   b. デスクトップ上の「WSL-VPNKit」ショートカットをダブルクリックします。
   c. コマンドプロンプトウィンドウが開き、wsl-vpnkitが起動します。このウィンドウは開いたままにしておきます。

5. Dockerイメージのpull (ローカルPC、WSL2のUbuntu):
   a. Ubuntu端末を開きます（まだ開いていない場合）。
   b. 以下のコマンドを実行して、Jupyter SciPy Notebookイメージをpullします：
      ```
      docker pull jupyter/scipy-notebook:latest
      ```
   c. ダウンロードの進行状況が表示されます。完了するまで待ちます。

6. pullしたイメージの確認 (ローカルPC、WSL2のUbuntu):
   a. 以下のコマンドを実行して、pullしたイメージを確認します：
      ```
      docker images
      ```
   b. 出力に「jupyter/scipy-notebook」が表示されていることを確認します。

7. (オプション) イメージの詳細情報の確認:
   a. 以下のコマンドを実行して、pullしたイメージの詳細情報を確認できます：
      ```
      docker inspect jupyter/scipy-notebook:latest
      ```

注意事項:
- snapを使用してDockerをインストールする場合、従来のapt-getを使用する方法と比べて若干の違いがある場合があります。
- イメージのpullには時間がかかる場合があります。ネットワーク速度によっては数分から数十分かかることがあります。
- 「latest」タグは最新版を示しますが、特定のバージョンを指定したい場合は、「latest」の代わりに具体的なバージョン番号を指定できます。

これらの手順により、snapを使用してDockerをインストールし、Jupyter SciPy Notebookイメージをpullすることができます。

Docker pullの手順

1. Dockerのインストール (ローカルPC、WSL2のUbuntu):
   a. Windowsのスタートメニューから「Ubuntu」を起動します。
   b. 以下のコマンドを順番に実行し、Dockerをインストールします：
      ```
      sudo apt update
      sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
      sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
      sudo apt update
      sudo apt install -y docker-ce
      ```
   c. Dockerサービスを開始し、自動起動を有効にします：
      ```
      sudo service docker start
      sudo systemctl enable docker
      ```
   d. 現在のユーザーをdockerグループに追加し、再ログインします：
      ```
      sudo usermod -aG docker $USER
      exit
      ```
   e. Ubuntu端末を再度開きます。

2. wsl-vpnkitの起動 (ローカルPC):
   a. VPNに接続していることを確認します。
   b. デスクトップ上の「WSL-VPNKit」ショートカットをダブルクリックします。
   c. コマンドプロンプトウィンドウが開き、wsl-vpnkitが起動します。このウィンドウは開いたままにしておきます。

3. Dockerイメージのpull (ローカルPC、WSL2のUbuntu):
   a. Ubuntu端末を開きます（まだ開いていない場合）。
   b. 以下のコマンドを実行して、Jupyter SciPy Notebookイメージをpullします：
      ```
      docker pull jupyter/scipy-notebook:latest
      ```
   c. ダウンロードの進行状況が表示されます。完了するまで待ちます。

4. pullしたイメージの確認 (ローカルPC、WSL2のUbuntu):
   a. 以下のコマンドを実行して、pullしたイメージを確認します：
      ```
      docker images
      ```
   b. 出力に「jupyter/scipy-notebook」が表示されていることを確認します。

5. (オプション) イメージの詳細情報の確認:
   a. 以下のコマンドを実行して、pullしたイメージの詳細情報を確認できます：
      ```
      docker inspect jupyter/scipy-notebook:latest
      ```

注意事項:
- イメージのpullには時間がかかる場合があります。ネットワーク速度によっては数分から数十分かかることがあります。
- 「latest」タグは最新版を示しますが、特定のバージョンを指定したい場合は、「latest」の代わりに具体的なバージョン番号を指定できます。

これらの手順により、Docker上でJupyter SciPy Notebookイメージがpullされ、使用可能な状態になります。

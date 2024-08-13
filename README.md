Docker loadの手順

1. DANZ02への接続 (ローカルPC、PowerShell):
   a. PowerShellを開きます。
   b. 以下のコマンドを実行して、DANZ02にSSH接続します：
      ```
      ssh danz02
      ```
   c. 必要に応じてパスワードを入力します。

2. アップロードしたイメージファイルの確認 (DANZ02、SSH接続したターミナル):
   a. 以下のコマンドを実行して、アップロードしたtarファイルの存在を確認します：
      ```
      ls -lh ~/scipy-notebook.tar
      ```
   b. ファイルのサイズと所有者情報が表示されることを確認します。

3. Dockerの状態確認 (DANZ02、SSH接続したターミナル):
   a. 以下のコマンドを実行して、Dockerデーモンが動作していることを確認します：
      ```
      sudo systemctl status docker
      ```
   b. Dockerが動作していない場合は、以下のコマンドで起動します：
      ```
      sudo systemctl start docker
      ```

4. Docker loadの実行 (DANZ02、SSH接続したターミナル):
   a. 以下のコマンドを実行して、tarファイルからDockerイメージをロードします：
      ```
      sudo docker load < ~/scipy-notebook.tar
      ```
   b. ロードの進行状況が表示されます。完了するまで待ちます。

5. ロードの確認 (DANZ02、SSH接続したターミナル):
   a. 以下のコマンドを実行して、ロードされたイメージを確認します：
      ```
      sudo docker images
      ```
   b. 出力に「jupyter/scipy-notebook」が表示されていることを確認します。

6. (オプション) イメージの詳細情報の確認 (DANZ02、SSH接続したターミナル):
   a. 以下のコマンドを実行して、ロードしたイメージの詳細情報を確認できます：
      ```
      sudo docker inspect jupyter/scipy-notebook:latest
      ```

7. クリーンアップ (DANZ02、SSH接続したターミナル):
   a. イメージのロードが完了したら、不要になったtarファイルを削除できます：
      ```
      rm ~/scipy-notebook.tar
      ```

注意事項:
- Docker loadの処理には時間がかかる場合があります。大きなイメージの場合、数分から数十分かかることがあります。
- DANZ02のストレージ容量が十分であることを確認してください。

トラブルシューティング:
- Docker loadでエラーが発生する場合、以下を確認してください：
  1. tarファイルが破損していないこと
  2. Dockerデーモンが正常に動作していること
  3. ユーザーがdockerグループに属していること（sudo不要の場合）

これらの手順により、アップロードしたDockerイメージのtarファイルがDANZ02上でロードされ、使用可能な状態になります。

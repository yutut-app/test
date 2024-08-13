wsl-vpnkitのインストールと設定手順

1. wsl-vpnkitのダウンロード (ローカルPC、ブラウザ):
   a. ブラウザで以下のURLにアクセスします：
      https://github.com/sakai135/wsl-vpnkit/releases/latest
   b. ページ下部の「Assets」セクションから「wsl-vpnkit.tar.gz」をクリックしてダウンロードします。
   c. ダウンロードしたファイルの場所を覚えておきます（通常はダウンロードフォルダ）。

2. wsl-vpnkitのインポート (ローカルPC、PowerShell):
   a. PowerShellを管理者として開きます（スタートメニューで「PowerShell」を右クリックし、「管理者として実行」を選択）。
   b. 以下のコマンドを実行して、ダウンロードしたファイルのあるディレクトリに移動します：
      ```
      cd C:\Users\YourUsername\Downloads
      ```
      注: 'YourUsername'を実際のWindowsユーザー名に置き換えてください。
   c. 次のコマンドを実行してwsl-vpnkitをインポートします：
      ```
      wsl --import wsl-vpnkit --version 2 $env:USERPROFILE\wsl-vpnkit wsl-vpnkit.tar.gz
      ```

3. wsl-vpnkitの起動用ショートカットの作成 (ローカルPC、デスクトップ):
   a. デスクトップ上で右クリックし、「新規作成」→「ショートカット」を選択します。
   b. 「項目の場所」に以下を入力します：
      ```
      C:\Windows\System32\wsl.exe -d wsl-vpnkit --cd /app wsl-vpnkit
      ```
   c. 「次へ」をクリックします。
   d. ショートカットの名前（例：「WSL-VPNKit」）を入力し、「完了」をクリックします。

4. wsl-vpnkitの動作確認 (ローカルPC):
   a. VPN接続を確立します（会社のVPNに接続）。
   b. 作成したデスクトップショートカット「WSL-VPNKit」をダブルクリックします。
   c. コマンドプロンプトウィンドウが開き、wsl-vpnkitが起動します。このウィンドウは開いたままにしておきます。

5. WSL2でのインターネット接続確認 (ローカルPC、Ubuntu on WSL2):
   a. Windowsのスタートメニューから「Ubuntu」を起動します。
   b. Ubuntu端末で以下のコマンドを実行してインターネット接続を確認します：
      ```
      ping 8.8.8.8
      ```
   c. 応答が返ってくれば、WSL2からインターネットに接続できています。

注意事項:
- wsl-vpnkitを使用する際は、毎回VPNに接続した後でデスクトップショートカットを実行する必要があります。
- wsl-vpnkitを終了する場合は、起動時に開いたコマンドプロンプトウィンドウを閉じます。

これらの手順により、wsl-vpnkitがインストールされ、VPN接続時にWSL2からインターネットにアクセスできるようになります。

WSL2のインストールと設定手順

1. WSL2の有効化 (ローカルPC、PowerShell管理者モード):
   a. Windowsのスタートメニューを右クリックし、「Windows PowerShell (管理者)」を選択します。
   b. 以下のコマンドを実行してWSLを有効化します：
      ```
      dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
      ```
   c. 続いて、仮想マシンプラットフォームを有効化します：
      ```
      dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
      ```
   d. コンピューターを再起動します。

2. WSL2 Linuxカーネル更新プログラムのインストール (ローカルPC、ブラウザ):
   a. 以下のMicrosoftのダウンロードページにアクセスします：
      https://aka.ms/wsl2kernel
   b. 「x64 マシン用 WSL2 Linux カーネル更新プログラム パッケージ」をダウンロードします。
   c. ダウンロードしたファイルを実行し、インストールを完了します。

3. WSL2をデフォルトバージョンとして設定 (ローカルPC、PowerShell):
   a. PowerShellを開きます（管理者権限は不要）。
   b. 以下のコマンドを実行します：
      ```
      wsl --set-default-version 2
      ```

4. Linuxディストリビューションのインストール (ローカルPC、Microsoft Store):
   a. Microsoft Storeを開きます。
   b. 検索バーに「Ubuntu」と入力します。
   c. 「Ubuntu」（バージョン番号なし）を選択し、「入手」をクリックしてインストールします。

5. Ubuntuの初期設定 (ローカルPC、Ubuntu):
   a. インストールが完了したら、Windowsのスタートメニューから「Ubuntu」を起動します。
   b. 初回起動時、新しいUNIXユーザー名とパスワードを設定するよう求められます。
      例：
      ユーザー名: yourusername
      パスワード: (任意の安全なパスワード)
   c. パスワードを入力する際、画面上には何も表示されませんが、正常に入力されています。

6. WSLのバージョン確認 (ローカルPC、PowerShell):
   a. PowerShellを開きます。
   b. 以下のコマンドを実行してインストールされたディストリビューションとそのバージョンを確認します：
      ```
      wsl --list --verbose
      ```
   c. 出力で、インストールしたUbuntuのVERSIONが2になっていることを確認します。

これらの手順により、WSL2とUbuntuがローカルPCにインストールされ、使用可能な状態になります。次の手順に進む前に、WSL2が正常にインストールされ、動作していることを確認してください。

何か質問や不明点があれば、お気軽にお聞きください。

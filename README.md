`/etc/apt/sources.list.d/nvidia-container-toolkit.list`に誤ってHTMLファイルの内容が書き込まれた可能性があります。このエラーを解決するためには、まずそのファイルの内容を確認し、正しいリポジトリ情報に修正する必要があります。以下の手順で対応します。

### 1. `nvidia-container-toolkit.list`の内容を確認
まず、`nvidia-container-toolkit.list`の内容を確認します。

```bash
cat /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

おそらくHTMLファイルのような内容が表示されるはずです。

### 2. `nvidia-container-toolkit.list`の内容を削除
次に、このファイルの内容を削除します。

```bash
sudo rm /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 3. 正しいリポジトリ情報を再追加
削除した後、正しいリポジトリを再度追加します。

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 4. パッケージリストの更新
再度、パッケージリストを更新します。

```bash
sudo apt-get update
```

これで問題が解決し、次に進めるはずです。




以下に、指定されたすべての手順を修正しながら説明します。

### 1. 依存パッケージのインストール
まず、必要なディレクトリを作成し、依存パッケージをインストールします。

```bash
sudo mkdir -p /etc/apt/keyrings
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
```

### 2. DockerのGPGキーを追加
次に、DockerのGPGキーを追加します。

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

### 3. Dockerのリポジトリを追加
リポジトリを追加する際、間違いがあったので修正しました。以下のように正しいリポジトリを追加してください。

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 4. パッケージリストの更新
リポジトリを追加後、パッケージリストを更新します。

```bash
sudo apt-get update
```

### 5. Dockerのインストール
Docker本体と関連パッケージをインストールします。

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io -y
```

### 6. ユーザーをDockerグループに追加
`<>`には、実際のユーザー名を入力してください。

```bash
sudo gpasswd -a <your-username> docker
```

これで、現在のユーザーがDockerコマンドをパスワードなしで実行できるようになります。

### 7. Dockerサービスのセットアップ
Dockerサービスの設定をリロードし、再起動します。

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 8. NVIDIA Dockerのインストール
次に、NVIDIAコンテナツールキットのGPGキーを追加します。

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

### 9. NVIDIA Dockerリポジトリの追加
NVIDIA Dockerのリポジトリを追加します。ここでも軽微な修正を加えています。

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 10. パッケージリストの更新
再度、パッケージリストを更新します。

```bash
sudo apt -y update
```

### 11. NVIDIA Dockerのインストール
次に、NVIDIAコンテナツールキットと関連パッケージをインストールします。

```bash
sudo apt-get install -y nvidia-container-runtime nvidia-container-toolkit nvidia-docker2
```

### 12. Dockerデーモンのリロード
最後に、Dockerデーモンをリロードして、変更を反映させます。

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

これでDockerとNVIDIA Dockerのセットアップは完了です。`docker run --gpus all`を使って、GPUを活用したコンテナを実行できるようになります。

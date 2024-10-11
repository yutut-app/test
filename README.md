### 1. 依存パッケージのインストール
まず、必要なディレクトリを作成し、依存パッケージをインストールする。

```bash
sudo mkdir -p /etc/apt/keyrings
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
```

### 2. DockerのGPGキーを追加
DockerのGPGキーを追加する。

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

### 3. Dockerリポジトリの追加
リポジトリを追加する。ここでは正確なリポジトリを使うようにしている。

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 4. パッケージリストの更新
リポジトリを追加した後、パッケージリストを更新する。

```bash
sudo apt-get update
```

### 5. Dockerのインストール
Dockerの本体と関連パッケージをインストールする。

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io -y
```

### 6. Dockerグループにユーザーを追加
まず、自分のユーザー名を確認するため、次のコマンドを実行する。

```bash
whoami
```

確認したユーザー名を使って、以下のコマンドでDockerグループに追加する。

```bash
sudo gpasswd -a <your-username> docker
```

ここで `<your-username>` には `whoami` で確認したユーザー名を入力する。

### 7. Dockerサービスのセットアップ
Dockerサービスの設定をリロードし、再起動する。

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 8. NVIDIA Dockerのインストール
次に、NVIDIAコンテナツールキットのGPGキーを追加する。

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

### 9. NVIDIA Dockerリポジトリの追加
NVIDIA Dockerのリポジトリを追加する。

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 10. パッケージリストの更新
再度、パッケージリストを更新する。

```bash
sudo apt -y update
```

### 11. NVIDIA Dockerのインストール
NVIDIAコンテナツールキットと関連パッケージをインストールする。

```bash
sudo apt-get install -y nvidia-container-runtime nvidia-container-toolkit nvidia-docker2
```

### 12. Dockerデーモンのリロード
最後に、Dockerデーモンをリロードし、変更を反映させる。

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

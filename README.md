DockerとNVIDIA Dockerのインストール手順を修正しつつ、以下にまとめました。間違いがあった部分を修正しています。

### 1. 必要なパッケージのインストール
まず、Dockerのインストールに必要なパッケージをインストールします。

```bash
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
```

### 2. Docker GPGキーの取得
次に、Dockerの公式GPGキーを取得します。

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

### 3. Dockerリポジトリの追加
次に、Dockerリポジトリを追加します。

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 4. Dockerのインストール
リポジトリを追加したら、パッケージリストを更新し、Dockerをインストールします。

```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io -y
```

### 5. ユーザーを`docker`グループに追加
現在のユーザーをDockerグループに追加して、パスワードなしでDockerを実行できるようにします。`<user>`の部分を実際のユーザー名に置き換えてください。

```bash
sudo usermod -aG docker $USER
```

### 6. Dockerサービスの設定と再起動
Dockerの設定ファイルを作成し、システムデーモンをリロードしてDockerを再起動します。

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 7. NVIDIA Dockerのインストール
次に、NVIDIA Container Toolkitをインストールします。

#### GPGキーの取得
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

#### リポジトリの設定
リポジトリの設定を行います。

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/libnvidia-container.list | \
sudo sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

#### パッケージリストの更新とNVIDIA Dockerのインストール
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-runtime nvidia-container-toolkit nvidia-docker2
```

### 8. Dockerの再起動
最後に、Dockerデーモンをリロードし、再起動します。

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

これで、DockerとNVIDIA Dockerのインストールが完了です。インストール後、以下のコマンドでNVIDIA Dockerが正しく設定されているか確認できます。

```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

エラーが出た場合や問題があれば教えてください。

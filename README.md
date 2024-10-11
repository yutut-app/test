以下に、参考にされた手順の正しいコマンドを修正してまとめました。

### 1. 必要なパッケージのインストール

まず、Dockerの依存パッケージをインストールします。

```bash
sudo mkdir -p /etc/apt/keyrings
sudo apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg-agent \
  software-properties-common
```

### 2. Docker GPGキーの追加

次に、DockerのGPGキーを取得し、APTの信頼キーとして設定します。

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

### 3. Dockerリポジトリの追加

Dockerの公式リポジトリをAPTソースリストに追加します。

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 4. パッケージリストの更新とDockerのインストール

リポジトリを追加した後、パッケージリストを更新し、Dockerをインストールします。

```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
```

### 5. Dockerグループへのユーザー追加

Dockerコマンドを非rootユーザーで実行するため、ユーザーを`docker`グループに追加します（`<ユーザー名>`はあなたのユーザー名に置き換えてください）。

```bash
sudo usermod -aG docker <ユーザー名>
```

### 6. Dockerの設定と再起動

Dockerサービスの設定ディレクトリを作成し、サービスを再読み込みして再起動します。

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 7. NVIDIA Dockerのインストール

NVIDIA Dockerを使用してGPUをコンテナで利用できるようにするため、次の手順でインストールします。

#### NVIDIAのGPGキーを追加

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

#### NVIDIAコンテナツールキットリポジトリの追加

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/libnvidia-container.list | sudo sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 8. パッケージリストの更新とNVIDIA Dockerのインストール

リポジトリを追加したら、パッケージリストを更新し、NVIDIA関連ツールをインストールします。

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-runtime nvidia-container-toolkit nvidia-docker2
```

### 9. Dockerサービスの再起動

最後に、Dockerサービスを再起動して変更を反映させます。

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

これで、DockerとNVIDIA Dockerがインストールされ、GPU対応のコンテナを動かせる環境が整いました。

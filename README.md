以下の手順に従って、Docker のインストールを進めてください。示された手順にはいくつかの誤字が含まれていますので、それらを修正しながら説明します。

### 1. 必要なパッケージのインストール

まず、Docker インストールに必要なパッケージをインストールします。`apt-transport-https` などが含まれています。

```bash
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
```

### 2. Docker の GPG キーを追加

次に、Docker の GPG キーを追加します。最新のキーを追加し、認証に使用します。

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

### 3. Docker のリポジトリを設定

リポジトリを追加して、Docker のパッケージを取得できるようにします。リポジトリのパスに誤りがあったため、修正したものを使用します。

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 4. パッケージリストの更新

Docker リポジトリを追加した後、パッケージリストを更新します。

```bash
sudo apt-get update
```

### 5. Docker のインストール

次に、Docker CE（Community Edition）、CLI、および containerd をインストールします。

```bash
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
```

### 6. Docker グループにユーザーを追加

`sudo` なしで Docker コマンドを実行できるように、現在のユーザーを Docker グループに追加します。`<>` にはユーザー名を入れてください。通常は `$(whoami)` で現在のユーザーを取得できます。

```bash
sudo gpasswd -a $(whoami) docker
```

その後、セッションを再起動する必要があります。

```bash
newgrp docker
```

### 7. Docker サービスの設定と再起動

Docker のサービスを再起動して有効にします。

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 8. NVIDIA Container Toolkit のインストール（オプション）

もし GPU を使用するために NVIDIA のサポートが必要であれば、次の手順で NVIDIA コンテナツールキットをインストールできます。

#### (1) NVIDIA GPG キーの追加

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

#### (2) NVIDIA コンテナツールキットのリポジトリの追加

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/$(arch)/nvidia-container-toolkit.list | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

#### (3) パッケージリストの更新

```bash
sudo apt-get update
```

#### (4) NVIDIA Docker のインストール

```bash
sudo apt-get install -y nvidia-container-runtime nvidia-container-toolkit nvidia-docker2
```

#### (5) Docker の再起動

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 9. Docker の動作確認

最後に、Docker のインストールが正常に行われたか確認するために、以下のコマンドを実行してみてください。

```bash
docker --version
docker run hello-world
```

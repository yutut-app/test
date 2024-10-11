以下に、指定された内容をすべて反映した手順を示す。

### 1. `GRUB`設定の変更
まず、GRUBの設定を変更して、Cgroupの設定を無効化する。

```bash
sudo nano /etc/default/grub
```

`GRUB_CMDLINE_LINUX_DEFAULT` の行を次のように変更する。

```bash
GRUB_CMDLINE_LINUX_DEFAULT="systemd.unified_cgroup_hierarchy=false"
```

変更を保存したら、GRUBの設定を更新して反映させる。

```bash
sudo update-grub
```

### 2. Cgroupドライバの変更
次に、`containerd`の設定ファイルを作成し、Cgroupの設定を変更する。

#### `containerd`のデフォルト設定をファイルに出力
```bash
containerd config default > config.toml
```

生成された `config.toml` ファイルを編集する。

```bash
sudo nano config.toml
```

`SystemdCgroup` を `false` から `true` に変更する。

```toml
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
  SystemdCgroup = true
```

変更後、この設定を `/etc/containerd/config.toml` に適用する。

```bash
cat config.toml | sudo tee /etc/containerd/config.toml
```

次に、`containerd`サービスを再起動する。

```bash
sudo systemctl restart containerd
```

### 3. デフォルトのコンテナランタイムの設定
`Docker`の設定ファイルを編集し、NVIDIAランタイムをデフォルトに設定する。

```bash
sudo nano /etc/docker/daemon.json
```

次の内容を追記する。既に設定がある場合は、ランタイム部分のみを追記する。

```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

変更後、`Docker`サービスを再起動する。

```bash
sudo systemctl restart docker
```

### 4. containerdの設定変更
最後に、`containerd`の設定ファイルを編集する。まず、`/etc/containerd/config.toml` を開く。

```bash
sudo nano /etc/containerd/config.toml
```

#### 9行目付近に次を追記
```toml
version = 2
```

#### デフォルトランタイムの変更
次に、`runc`を`nvidia`に変更する。

```toml
[plugins."io.containerd.grpc.v1.cri".containerd]
  default_runtime_name = "nvidia"
```

#### NVIDIAランタイムの設定を追加
ランタイム設定の次行に、次の内容を追加する。

```toml
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  privileged_without_host_devices = false
  runtime_engine = ""
  runtime_root = ""
  runtime_type = "io.containerd.runc.v2"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
    BinaryName = "/usr/bin/nvidia-container-runtime"
```

変更後、再度`containerd`サービスを再起動する。

```bash
sudo systemctl restart containerd
```

これで、Cgroupの設定変更とNVIDIAランタイムの設定が完了した。

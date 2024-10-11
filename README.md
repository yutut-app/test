以下に、これまでの内容を踏まえて、GPU対応のコンテナイメージを選択し、VSCode上で.ipynbファイルをGPUを使用して実行できる手順をまとめた。

### 1. 適切なGPU対応のコンテナイメージの選択

VSCode上で事前学習や転移学習を実装できる最適なGPU対応のコンテナイメージとして、**TensorFlowのGPU対応コンテナ**が推奨される。

使用するコンテナイメージ:
```bash
tensorflow/tensorflow:latest-gpu-jupyter
```

このイメージには、Jupyter Notebookサポートが組み込まれており、TensorFlowを用いた事前学習や転移学習に最適。

### 2. DockerでTensorFlowのGPUコンテナを起動

次に、DockerでTensorFlowのGPU対応コンテナを起動する。ノートブックファイルがあるディレクトリを正しく指定し、GPUを有効化する。

以下のコマンドでコンテナを実行する。`<path_to_your_notebooks>` は実際のノートブックのパスに置き換える。

```bash
sudo docker run --gpus all -it --rm -p 8888:8888 -v <path_to_your_notebooks>:/tf/notebooks tensorflow/tensorflow:latest-gpu-jupyter
```

#### コマンドの説明:
- `--gpus all`: GPUを利用可能にするオプション。
- `-it`: インタラクティブモードでコンテナを実行。
- `--rm`: コンテナ終了時に自動で削除。
- `-p 8888:8888`: ホストのポート8888をコンテナのポート8888にマッピング（Jupyter Notebook用）。
- `-v <path_to_your_notebooks>:/tf/notebooks`: ホストマシンのノートブックディレクトリをコンテナ内にマウント。

### 3. Jupyter Notebookにアクセス

コンテナが起動すると、Jupyter NotebookのURLがコンテナログに表示される。次のようなURLが表示される。

```
To access the notebook, open this file in a browser:
    http://127.0.0.1:8888/?token=<your_token>
```

ブラウザでこのURLを開くことで、Jupyter Notebookにアクセスできる。

### 4. VSCodeで.ipynbファイルを開く

VSCode上で.ipynbファイルを開いて、Jupyterサーバーに接続する手順は以下の通り。

1. **Jupyter拡張機能を確認**
   - VSCodeにJupyter拡張機能がインストールされていることを確認する。

2. **.ipynbファイルをVSCodeで開く**
   - VSCodeで実行したい.ipynbファイルを開く。

3. **Jupyterサーバーを手動で指定**
   - VSCodeの上部に表示されている「カーネルの選択」から、「Jupyterサーバーの接続先を指定」を選択。
   - 表示された入力欄に、コンテナ起動時に表示された `http://127.0.0.1:8888/?token=<your_token>` のURLを入力。

4. **接続の確認**
   - 正しく接続されると、VSCode上で直接.ipynbファイルを実行できるようになる。

### 5. GPUが使用されているか確認

VSCode上でGPUが正しく使用されているか確認するために、以下のコードを実行する。

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

`Num GPUs Available: 1` 以上が表示されれば、GPUが正しく認識されている。

### 6. GPU使用のテスト

GPUが実際に計算に使用されているか確認するために、次のコードを実行する。

```python
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

# GPU上での行列演算を試す
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)

print(c)
```

ログに `Executing op` とともに `/GPU:0` が表示されれば、GPUで計算が行われていることになる。

この手順で、VSCode上でGPUを使用したJupyter Notebook (.ipynbファイル) の実行環境が整う。

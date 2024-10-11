sudo docker run --gpus all -it --rm -p 8888:8888 -v ~/path_to_your_notebooks:/tf/notebooks tensorflow/tensorflow:latest-gpu-jupyter --allow-root



次に進むために、まず適切なGPU対応のコンテナイメージを選択し、VSCode上で.ipynbファイルをGPUを使用して実行できるようにする手順を説明する。

### 1. GPU対応のコンテナイメージの選択

VSCode上で事前学習および転移学習が実装できる最適なGPU対応のコンテナイメージとして、TensorFlowまたはPyTorchが含まれる公式のイメージが推奨される。

ここでは、**TensorFlowのGPU対応コンテナ**を使用する。これは、事前学習、転移学習の両方に対応しており、Jupyter Notebookサポートも組み込まれている。

#### イメージの選択:
```bash
tensorflow/tensorflow:latest-gpu-jupyter
```

### 2. DockerでTensorFlowのGPUコンテナを起動

次に、Docker上でGPU対応のTensorFlowコンテナを起動する。以下のコマンドを実行して、Jupyter Notebook付きのGPU対応TensorFlowコンテナを立ち上げる。

```bash
sudo docker run --gpus all -it --rm -p 8888:8888 -v ~/path_to_your_notebooks:/tf/notebooks tensorflow/tensorflow:latest-gpu-jupyter
```

#### コマンド説明:
- `--gpus all`: GPUを使用可能にする
- `-it`: インタラクティブモードでの起動
- `--rm`: コンテナ終了後に自動で削除
- `-p 8888:8888`: ホストのポート8888をコンテナのポート8888にバインド（Jupyter Notebook用）
- `-v ~/path_to_your_notebooks:/tf/notebooks`: ローカルのノートブックフォルダをコンテナ内にマウント

`~/path_to_your_notebooks` の部分を、実際のノートブックファイルが保存されているディレクトリに置き換える。

### 3. コンテナ内のJupyter Notebookへのアクセス
コンテナが立ち上がると、Jupyter Notebookが起動し、以下のような出力が表示される。

```
To access the notebook, open this file in a browser:
    http://localhost:8888/?token=<your_token>
```

表示されたURLをブラウザにコピー＆ペーストして、Jupyter Notebookにアクセスできる。

### 4. VSCodeで.ipynbファイルを開く
VSCodeからもJupyter Notebookにアクセスするために、以下の手順を行う。

1. VSCodeの左側のサイドバーで、**Jupyter** 拡張機能をインストール。
2. VSCodeの**コマンドパレット** (`Ctrl + Shift + P`) を開き、「**Jupyter: Specify Jupyter Server for Connections**」を選択する。
3. 手動で `http://localhost:8888/?token=<your_token>` の形式で、JupyterサーバーのURLを入力。

これで、VSCode上で直接Jupyter Notebook (.ipynbファイル) を開いて、GPUを使用して実行できるようになる。

### 5. GPUが使用されているか確認

最後に、.ipynbファイル内でGPUが使用されているかを確認するために、次のコードを実行する。

#### TensorFlowの場合:
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

`Num GPUs Available: 1` 以上が表示されれば、GPUが正しく認識されている。

これで、VSCode上でGPUを使用して.ipynbファイルが実行できる環境が整った。

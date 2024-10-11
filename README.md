GPU対応のコンテナイメージを使って、Jupyter Notebook上でCNNや転移学習を実行できる環境を構築する手順を示す。

### 1. TensorFlowのGPU対応イメージを使用する

TensorFlowやPyTorchなど、一般的なディープラーニングフレームワークの公式Dockerイメージを使うと、GPUを利用したモデルのトレーニングが可能になる。ここでは、`tensorflow/tensorflow:latest-gpu-jupyter` イメージを使用して、.ipynbファイルでCNNや転移学習を実装できる環境を構築する。

#### TensorFlowのGPU対応Jupyterイメージをプルする
以下のコマンドで、TensorFlowの最新のGPU対応Jupyterイメージを取得する。

```bash
sudo docker pull tensorflow/tensorflow:latest-gpu-jupyter
```

### 2. コンテナの起動
Jupyter Notebookを使用するため、ポートを指定してコンテナを起動する。また、`--gpus all` オプションで、GPUを利用可能な設定にする。

#### コンテナを起動するコマンド
以下のコマンドで、TensorFlowのコンテナを起動する。

```bash
sudo docker run --gpus all -it --rm -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
```

- `--gpus all`: すべてのGPUを使用可能にするオプション
- `-it`: インタラクティブモードでコンテナを起動
- `--rm`: コンテナ終了時に自動で削除
- `-p 8888:8888`: ホストとコンテナのポート8888を接続し、Jupyter Notebookにアクセス可能にする

### 3. Jupyter Notebookにアクセス
コンテナが起動すると、Jupyter Notebookのアクセス用URLが表示される。URLは、以下のような形式になっている。

```
http://localhost:8888/?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

このURLをブラウザに貼り付けることで、Jupyter Notebookにアクセスできる。

### 4. Jupyter Notebookでの作業
Jupyter Notebook上で、`.ipynb` ファイルを開き、CNNや転移学習の実装を行う。TensorFlowのGPUサポートが有効になっているため、GPUを活用して効率的にモデルをトレーニングできる。

#### GPUが認識されているかの確認
以下のコードをJupyter Notebookで実行して、TensorFlowがGPUを認識しているか確認できる。

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

`Num GPUs Available` の出力が `1` 以上であれば、GPUが正常に認識されている。

これで、GPU対応のコンテナイメージを使用した環境が構築され、Jupyter NotebookでCNNや転移学習が実装できる状態になる。

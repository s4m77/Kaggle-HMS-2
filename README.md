## Setup 

### Environment

1. Initial env setup

```shell
conda env create -f environment.yaml -y

conda activate graph
```

2. PyTorch setup: need to install the correct packages with the correct cpu/cuda wheel. In our case:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install torcheeg
```

### Dataset

1. Download the [HMS dataset](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification) from Kaggle (note: you will have to join the challenge and accept T&C in order to dowload)

```shell
cd data/raw

kaggle competitions download -c hms-harmful-brain-activity-classification
```

2. Run the preprocessing notebook

```shell
jupyter execute notebooks/eda.ipynb
```

3. Create the Graph Dataset

```shell
python src/data/make_graph_dataset.py
```

### Train the models

1. AlphaHMS:

```shell
python src/train.py --train-config configs/train.yaml
```

2. Baseline EEG

```shell
python src/train.py --train-config configs/train.yaml
```

3. Baseline MLP

```shell
python src/train_mlp.py --config configs/training_mlp.yaml
```

Training scripts use WandB as logger; you may be asked to log into your account beforehand.

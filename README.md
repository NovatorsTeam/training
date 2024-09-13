# hackaton_model_training

The repository for training the if-pallete-need-replacment classification model for AI Talent Hack.

# Installing dependencies
```bash
apt update && apt-get install ffmpeg libsm6 libxext6 -y
pip install pdm
pdm install
```

Further please use `pdm add` command to save dependencies in pyptoject.toml file. 

# Dataset structure

The dataset of pallets for image classification must have the structure of directories as listed below:
```
Processed
├── bottom
│   ├── replace
│   │   ├── 1.jpg
│   │   └── ...
│   └── remain
│       ├── 2.jpg
│       └── ...
└── side
    ├── replace
    │   ├── 3.jpg
    │   └── ...
    └── remain
        ├── 4.jpg
        └── ...
```

# Dataset Source
You can download our processed pallet images from [Goolge Drive](https://drive.google.com/drive/folders/1UPX0piYZj0Qi5x7uVTgp4FKtyOMyv2wi) 

# MLFlow settings

Rename the `.env.example` to `.env` and specify the **MLFLOW_EXPERIMENT_NAME** and **MLFLOW_TRACKING_URI** variables.
If you already set this variables as environment variables - you can skip this part.

# Training

To run train script run this command with specified settings
```bash
python -m src.hackaton_model_training.train \
--dataset_path=data/Processed \
--lr=0.001 \
--save_path=models \
--save_every=10 \
--epochs=100 \
--model_name=resnet \
--device=cuda \
--mlflow_tracking=True
```

# Testing

You can run tests by pytest library
```
pytest -v
```

# Authors

AITH Students:
- Alexey Laletin (alex.klark.laletin@yandex.com)
- George Kiselev (gsx2002@yandex.ru)
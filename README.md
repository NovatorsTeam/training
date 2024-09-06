# hackaton_model_training

The repository for training the if-pallete-need-replacment classification model for AI Talent Hack.

# Installing dependencies
```bash
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

# Testing

You can run tests by pytest library
```
pytest -v
```

# Authors

AITH Students:
- Alexey Laletin (alex.klark.laletin@yandex.com)
- George Kiselev (gsx2002@yandex.ru)
# TINY-ViT: Vision Transformer from Scratch

## Introduction
TINY-ViT offers a minimalist, yet complete implementation of the Vision Transformer (ViT) architecture for computer vision tasks. This project aims to provide a clear and structured approach to building Vision Transformers, making it accessible for educational purposes and practical applications alike.

## Features
- **Modular Design**: Clear separation of components like data processing, model architecture, and training routines.
- **Customizable**: Easy to adapt the architecture and data pipeline for various datasets and applications.
- **Poetry Dependency Management**: Utilizes Poetry for simple and reliable package management.

## Project Structure
```
TINY-VIT-TRANSFORMER-FROM-SCRATCH
│
├── dataset                   # Dataset directory
├── tests                     # Test scripts
├── tiny_vit_transformer_from_scratch
│   ├── core                  # Core configurations and caching
│   ├── data                  # Data processing modules
│   └── model                 # Transformer model components
├── train.py                  # Script to train the model
├── finetune.py               # Script for fine-tuning the model
├── README.md                 # Project README file
├── poetry.lock               # Poetry lock file for consistent builds
└── pyproject.toml            # Poetry project file with dependency descriptions
```

## Installation
To install the project and its dependencies:

```bash
poetry install
```

## Usage
### Training
To train the model using the default configuration:

```bash
poetry run python train.py
```

### Fine-Tuning
To fine-tune a pre-trained model:

```bash
poetry run python finetune.py
```

## Contributions
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Citation
Please cite this project if it helps your research. Sample BibTeX entry:

```bibtex
@misc{tiny_vit_2023,
  title={TINY-ViT: Vision Transformer from Scratch},
  author={Ben alla ismail},
  year={2023},
  url={https://github.com/yourusername/tiny-vit-transformer-from-scratch}
}
```

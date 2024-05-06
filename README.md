<div align="center">
  <img src="https://github.com/benisalla/Tiny-ViT-Transformer-from-scratch/assets/89405673/035598be-ea1c-4501-947a-ff51524e78ef" width="200" height="200"/>
  <h1>TINY-ViT: Vision Transformer from Scratch</h1>
  <p>Implementing a Vision Transformer model from the scratch.</p>
</div>







---







## Table of Contents 📘
- [About The Project](#about-the-project)
- [Built With](#built-with)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Fine-Tuning](#fine-tuning)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Contributors](#contributors)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [About Me](#about-me)




---



## About The Project

<div align="center">
  <img src="https://github.com/benisalla/Tiny-ViT-Transformer-from-scratch/assets/89405673/4935a979-a8e8-40f0-8ffb-e621025aac2f" width="600" height="300"/>
</div>

TINY-ViT offers a minimalist, yet complete implementation of the Vision Transformer (ViT) architecture for computer vision tasks. This project aims to provide a clear and structured approach to building Vision Transformers, making it accessible for educational purposes and practical applications alike.





---




## Features

- **Modular Design**: Clear separation of components like data processing, model architecture, and training routines.
- **Customizable**: Easy to adapt the architecture and data pipeline for various datasets and applications.
- **Poetry Dependency Management**: Utilizes Poetry for simple and reliable package management.
- **Advanced Embedding Techniques**: Implements three distinct techniques for image embedding in Vision Transformers:
  - **ViTConv2dEmbedding**: Utilizes a Conv2D layer to transform input images into a sequence of flattened 2D patches, with a learnable class token appended.
  - **ViTLNEmbedding**: Applies layer normalization to flattened input patches before projecting them into an embedding space, enhancing stability and performance.
  - **ViTPyCon2DEmbedding**: Offers a unique tensor reshaping strategy to transform input images into a sequence of embedded patches, also including a learnable class token.
- **Custom Activation Function**: Incorporates the **ViTGELUActFun** class, which implements the Gaussian Error Linear Unit (GELU), providing smoother gating behavior than traditional nonlinearities like ReLU.

---

### ViTConv2dEmbedding
```python
class ViTConv2dEmbedding(nn.Module):
```

### ViTLNEmbedding
```python
class ViTLNEmbedding(nn.Module):
```

### ViTPyCon2DEmbedding
```python
class ViTPyCon2DEmbedding(nn.Module):
```

### ViTGELUActFun Activation Function
```python
class ViTGELUActFun(nn.Module):
```














---





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





---






### Built With
This section should list any major frameworks/libraries used to bootstrap your project:
- [MyBest framework ever: PyTorch](https://pytorch.org/)

---





## Getting Started

To get a local copy up and running follow these simple steps.





### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/benisalla/tiny-vit-transformer-from-scratch.git
   ```
2. Install Poetry packages
   ```sh
   poetry install
   ```





---





## Usage

how you can use this code

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






---





## Model Performance

The tiny-vit model was evaluated on a comprehensive set of test images to gauge its accuracy and performance. Here are the results:

- **Accuracy on test images**: 81.60%

These results demonstrate the effectiveness of the tiny-vit model in handling complex image recognition tasks. We continuously seek to improve the model and update the metrics as new test results become available.

![image](https://github.com/benisalla/Tiny-ViT-Transformer-from-scratch/assets/89405673/62531c3f-6684-4000-a151-acee6a399ab3)

![image](https://github.com/benisalla/Tiny-ViT-Transformer-from-scratch/assets/89405673/7aafbd0a-f48b-46dd-9caf-d99f42e063e3)



---





## Roadmap

See the [open issues](https://github.com/benisalla/tiny-vit-transformer-from-scratch/issues) for a list of proposed features (and known issues).





---





## Contributing 

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.




---







## Contributors

- **Asmae El-Ghezzaz** - Data Scientist/ML
  - [GitHub](https://github.com/aelghezzaz)
  - Contributions: Provided expertise in machine learning and data science methodologies.

- **Idriss El Houari** - Data Scientist
  - [GitHub](https://github.com/idrisselhouari)
  - [Kaggle](https://www.kaggle.com/idrisselhouari)
  - Contributions: Curated the plant disease dataset and developed the initial analysis notebook.
  
- **Farheen Akhter** - Graduate Student at California State University
  - [GitHub](https://github.com/FarheenAkhter786)
  - [Kaggle](https://www.kaggle.com/feenuakhter)
  - Contributions: Worked on improving crop yield and pest/disease detection through data analytics. Provided dataset and analytical insights on local farms in Ghana.

- **Aicha Dessa** - Data Scientist Intern
  - [GitHub](https://github.com/aichadessa06)
  - [Kaggle](https://www.kaggle.com/aichadessa)
  - Contributions: Analyzed plant disease recognition data and contributed to model training and testing processes.
  
- **Zeroual Salma** - Student at Agronomic and Veterinary Institute Hassan II
  - [GitHub](https://github.com/salmazl)
  - Contributions: Focused on plant pathology and contributed to dataset analysis and insights into Botrytis disease.
  
- **El Fakir Chaimae** - Master's Student in Artificial Intelligence
  - [GitHub](https://github.com/chaimaeelfakir)
  - Contributions: Provided datasets on pathogen detection and collaborated on developing AI models for disease prediction.
  
- **Laghbissi Salma** - Master's Student in Software Engineering for Cloud Computing
  - [GitHub](https://github.com/salma-laghbissi)
  - Contributions: Researched on plant pathologies and contributed significantly to the dataset understanding and processing.



---







## Authors

- **Ismail Ben Alla (Me  😉)** - [View My GitHub Profile](https://github.com/benisalla)
- **Asmae El-Ghezzaz (a friend of mine)** - special thanks for her help and advices about the documentation




---







## Acknowledgements

This project owes its success to the invaluable support and resources provided by several individuals and organizations. A heartfelt thank you to:

- [**Asmae El-Ghezzaz**](https://www.linkedin.com/in/asmae-el-ghezzaz) - For inviting me to be a member of Moroccan Data Scientists (MDS), where I had the opportunity to develop this project.
- [**Moroccan Data Scientists (MDS)**](https://www.linkedin.com/company/moroccands) - Although I am no longer a member, I hold great admiration for the community and wish it continued success.
- **Pests and Vigitebles Diseased Detection Team in MDS** - Aicha, hiba, idriss, farheen, asmae, ...
- [**PyTorch**](https://pytorch.org/) - For the powerful and flexible deep learning platform that has made implementing models a smoother process.
- [**Kaggle**](https://www.kaggle.com/) - For providing the datasets used in training our models and hosting competitions that inspire our approaches.
- [**Google Colab**](https://colab.research.google.com/) - For the computational resources that have been instrumental in training and testing our models efficiently.






---







## License

This project is made available under **fair use guidelines**. While there is no formal license associated with the repository, users are encouraged to credit the source if they utilize or adapt the code in their work. This approach promotes ethical practices and contributions to the open-source community. For citation purposes, please use the following:

```bibtex
@misc{tiny_vit_2024,
  title={TINY-ViT: Vision Transformer from Scratch},
  author={Ben Alla Ismail},
  year={2024},
  url={https://github.com/benisalla/tiny-vit-transformer-from-scratch}
}
```





---





## About Me

🎓 **Ismail Ben Alla** - Neural Network Enthusiast

I am deeply passionate about exploring artificial intelligence and its potential to solve complex problems and unravel the mysteries of our universe. My academic and professional journey is characterized by a commitment to learning and innovation in AI, deep learning, and machine learning.

### What Drives Me
- **Passion for AI**: Eager to push the boundaries of technology and discover new possibilities.
- **Continuous Learning**: Committed to staying informed and skilled in the latest advancements.
- **Optimism and Dedication**: Motivated by the challenges and opportunities that the future of AI holds.

I thoroughly enjoy what I do and am excited about the future of AI and machine learning. Let's connect and explore the endless possibilities of artificial intelligence together!


<div align="center">
  <a href="https://twitter.com/ismail_ben_alla" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="ismail_ben_alla" height="30" width="40" />
  </a>
  <a href="https://linkedin.com/in/ismail-ben-alla-7144b5221/" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="ismail-ben-alla-7144b5221/" height="30" width="40" />
  </a>
  <a href="https://instagram.com/ismail_ben_alla" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="ismail_ben_alla" height="30" width="40" />
  </a>
</div>



---


<div align="center">
  <h4>Get ready to see pixels transform into insights 🌟🔍✨</h4>
  <img src="https://github.com/benisalla/Tiny-ViT-Transformer-from-scratch/assets/89405673/087e0049-d113-4df6-8fb3-183ebc4f85e1" width="500" height="300"/>
</div>

# LAVA
The official pytorch implementation of the paper [LAVA: Label-efficient Visual Learning and Adaptation](https://arxiv.org/abs/2210.10317) (appearing in WACV2023)

<p align="center">
  <img class="center" src="https://github.com/islam-nassar/lava/blob/main/aux/LAVA_teaser.png" alt="LAVA Conceptual Diagram" width="500">
</p>

## Overview
LAVA is a transfer learning method combining self-supervised vision transformers, multi-crop pseudo-labeling, and weak supervision using language to enable transfer with limited labels to different visual domains. It provides a training recipe which achieves state-of-the-art results in semi-supervised transfer and few-shot cross-domain transfer. In what follows, we first, detail how to install LAVA dependencies, then we detail both semi-supervised and few-shot settings. 

## Install Dependencies

- Create a new environment and install LAVA dependencies using ```pip install -r requirements.txt``` 
- For running Few-shot experiments, you also need to install the meta-dataset dependencies as per their [installation instructions](https://github.com/google-research/meta-dataset#installation).  

## Semi-supervised Learning (SSL)
Our SSL transfer learning setup includes: 
1) self-supervised pretraining on the source dataset
2) self-supervised fine-tuning on the target dataset
3) supervised fine-tuning on the target dataset

First, we discuss how to prepare the SSL datasets then we detail how to run LAVA for each of the three stages.

### SSL Dataset preparation
We use a standard directory structure for all our datasets (source and target) to enable running experiments on any dataset of choice without the need to edit the dataloaders. The datasets directory follow the below structure (only shown for ImageNet but is the same for all other datasets):
```
datasets
└───ImageNet
   └───train
       └───labelled
           └───australian_terrier (class name for Imagenet can also be represented by its wordnet id e.g.: n02096294)  
             │   <australian_terrier_train_1>.jpeg
             │   <australian_terrier_train_2>.jpeg
             │   ...
           └───egyptian_cat (or n02124075)
             │   <egyptian_cat_train_1>.jpeg
             │   <egyptian_cat_train_2>.jpeg
             │   ...
           ...
           
       └───unlabelled
           └───UNKNOWN_CLASS (this directory can be empty if all the labels are used, i.e. in a fully supervised setting)
             │   <australian_terrier_train_xx>.jpeg
             │   <egyptian_cat_train_xx>.jpeg
             │   <badger_train_xx>.jpeg
             │   ...
             
   └───val
       └───australian_terrier
             │   <australian_terrier_val_1>.jpeg
             │   <australian_terrier_val_2>.jpeg
             │   ...
       └───egyptian_cat
             │   <egyptian_cat_val_1>.jpeg
             │   <egyptian_cat_val_2>.jpeg
             │   ...
           ...
```
To preprocess a generic dataset into the above format, you can refer to `utils/data_utils.py` for several examples.

### SSL training (stage 1 - source self-supervised pretraining)
### SSL training (stage 2 - target self-supervised fine-tuning)
### SSL training (stage 3 - target supervised fine-tuning)
### SSL Validation

## Few-shot Learning (FSL)
### FSL Dataset preparation
Follow meta-dataset to download and preprocess the meta-dataset (10 datasets)
Set env variables 

### FSL training and evaluation
For FSL, we encapsulate training and evaluation using `few_shot_runner.py`. You only need to specify the dataset and the number of episodes, then our launcher will take care of the rest. For example, if you need to run 600 few-shot episodes on `textures`, run:





## Citation

If you find our work useful, please consider giving a star and citing our work using below:

```
@article{nassar2022lava,
  title={LAVA: Label-efficient Visual Learning and Adaptation},
  author={Nassar, Islam and Hayat, Munawar and Abbasnejad, Ehsan and Rezatofighi, Hamid and Harandi, Mehrtash and Haffari, Gholamreza},
  journal={arXiv preprint arXiv:2210.10317},
  year={2022}
}
```

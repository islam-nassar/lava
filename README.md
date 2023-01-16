# LAVA
The pytorch implementation of the paper [LAVA: Label-efficient Visual Learning and Adaptation](https://arxiv.org/abs/2210.10317) (appearing in WACV2023)

<p align="center">
  <img class="center" src="https://github.com/islam-nassar/lava/blob/main/aux/LAVA_teaser.png" alt="LAVA Conceptual Diagram">
</p>

## Overview
LAVA is a transfer learning method combining self-supervised vision transformers, multi-crop pseudo-labeling, and weak supervision using language to enable transfer with limited labels to different visual domains. It provides a training recipe which achieves state-of-the-art results in semi-supervised transfer and few-shot cross-domain transfer. 
In what follows, we detail: 
- [Installing LAVA dependencies](https://github.com/islam-nassar/lava/blob/main/README.md#install-dependencies)
- [LAVA for Semi-supervised Learning](https://github.com/islam-nassar/lava/blob/main/README.md#semi-supervised-learning-ssl)
- [LAVA for Few-shot Learning](https://github.com/islam-nassar/lava/blob/main/README.md#few-shot-learning-fsl)

## Install Dependencies

- Create a new environment and install LAVA dependencies using ```pip install -r requirements.txt``` 
- For running Few-shot experiments, you also need to install the meta-dataset dependencies as per their [installation instructions](https://github.com/google-research/meta-dataset#installation).  

## LAVA for Semi-supervised Learning (SSL)
Our SSL transfer learning procedure includes: 
1) self-supervised pretraining on the source dataset
2) self-supervised fine-tuning on the target dataset
3) semi-supervised fine-tuning on the target dataset

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
           └───UNKNOWN_CLASS (this directory can be empty if all the labels are used, e.g. fully supervised setting)
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
...
```
To preprocess a generic dataset into the above format, you can refer to `data_utils.py` for several examples. (coming soon)

### SSL training (stage 1 - source self-supervised pretraining)
LAVA uses DINO method for self-supervised pretraining. Hence, one can use the pretrained models provided by [DINO](https://github.com/facebookresearch/dino#pretrained-models) or alternatively, can pretrain a new model using the provided script:

```
./scripts/lava_ssl_1_source_self_pretrain.sh
```
Notes:
- If you are to use a [pretrained DINO model](https://github.com/facebookresearch/dino#pretrained-models), make sure to use the full ckpt (full checkpoint) and not just the backbone because our method uses the head during target self-supervised fine-tuning.
- You can pretrain your own DINO model using LAVA's repo by invoking the above script. You should edit the header of the script to reflect your source dataset (we use ImageNet but you can use any dataset of your choice as long as it is in the described format)

The pretrained model we used in all our SSL experiments can be found here. ([lava_ssl_imagenet_vit_s_16](https://drive.google.com/file/d/1AiSYbKboq6RYrqB0LxjRIG4iK1yuV4aL/view?usp=sharing))

### SSL training (stage 2 - target self-supervised fine-tuning)
In the second stage, LAVA fine-tune the source self-supervised pretrained model to the unlabeled instances of the target dataset. You can achieve this by running:
```
./scripts/lava_ssl_2_target_self_finetune.sh
```
You must edit the header of the above script to reflect the target dataset directory and the source pretrained weights. Refer to our example which uses imagenet source and domain_net clipart as target.

### SSL training (stage 3 - target semisupervised finetuning)
Finally, in the third stage, LAVA fine-tunes the target self-supervised pretrained model to the labeled and unlabeled instances of the target dataset by employing supervised fine-tuning and multi-crop pseudo-labeling. You can achieve this by running:
```
./scripts/lava_ssl_3_target_semisup_finetune.sh
```
You must edit the header of the above script to reflect the target dataset directory and the pretrained weights. Refer to our example which uses imagenet finetuned on clipart pretrained weights and adapt to domain_net clipart with 4-shots (i.e. 4 images per class).
### SSL evaluation
To evaluate LAVA, you can use eval_linear.py or eval_knn.py which is the standalone versions of eval_or_predict_linear.py and eval_or_predict_knn.py


## LAVA for Few-shot Learning (FSL)
Our FSL transfer learning procedure includes: 
1) self-supervised pretraining on the source dataset (We use ImageNet train split from meta-dataset)
2) Language module pretraining on the source dataset
3) FSL episodes - semisupervised finetuning on the support set and evaluation on query set.

First, we discuss how to prepare the meta-dataset then we detail how to run LAVA for each of the three stages.

### FSL Dataset preparation
Follow meta-dataset repo to [download and preprocess the meta-dataset (10 datasets)](https://github.com/google-research/meta-dataset#downloading-and-converting-datasets). You shall end up with a directory which contains two subdirectories `RECORDS` and `SPLITS`. (e.g. /home/data/meta_dataset)

Set the following environment variable:
```
export METADATASET_ROOT=/home/data/meta_dataset
```

### FSL source pretraining
For source self-supervised pretraining in case of few-shot learning, we provide the following script which trains LAVA on ImageNet meta-dataset train classes (716 classes out of the 1000):
```
./scripts/lava_fsl_1_source_self_pretrain.sh
```

### FSL source language pretraining
Subsequently, you train the language adapter MLP while freezing LAVA's backbone model by running:
```
./scripts/lava_fsl_2_source_language_pretrain.sh
```

The pretrained model we used in all our FSL experiments can be found here. ([pretrained weights on ImageNet 716 classes with language MLP](https://drive.google.com/file/d/1GvGNLe0vO7fYn6EcxK7lOjYORaKa29G7/view?usp=sharing))

### FSL target training and evaluation (FSL episodes)
Finally, to evaluate LAVA on FSL meta-dataset episodes, we encapsulate training and evaluation using `fewshot_runner.py`. You only need to specify the dataset and the number of episodes, then our runner code will take care of the rest. We provide an example which runs 600 few-shot episodes on `mscoco` dataset:
```
./scripts/lava_fsl_3_target_finetune_episodes.sh
```

You can adjust the above script header to include any of the meta-datasets below and any number of episodes.

`meta datasets: 'aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012', 'omniglot', 'quickdraw', 'vgg_flower', 'traffic_sign', 'mscoco'`



## Citation

If you find our work useful, please consider giving a star and citing our work using below:

```
@inproceedings{nassar2023lava,
  title={LAVA: Label-efficient Visual Learning and Adaptation},
  author={Nassar, Islam and Hayat, Munawar and Abbasnejad, Ehsan and Rezatofighi, Hamid and Harandi, Mehrtash and Haffari, Gholamreza},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={147--156},
  year={2023}
}
```

## Acknowledgement

This repo is partially based on the [DINO github repo](https://github.com/facebookresearch/dino). 

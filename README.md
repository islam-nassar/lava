# LAVA
The official pytorch implementation of the paper [LAVA: Label-efficient Visual Learning and Adaptation](https://arxiv.org/abs/2210.10317) (appearing in WACV2023)

<p align="center">
  <img class="center" src="https://github.com/islam-nassar/lava/blob/main/LAVA_teaser.pdf" alt="SemCo Conceptual Diagram" width="500">
</p>

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

## Install Dependencies

- Create a new environment and install dependencies using ```pip install -r requirements.txt``` 


## Dataset
We use a standard directory structure for all our datasets to enable running the code on any dataset of choice without the need to edit the dataloaders. The datasets directory follow the below structure (only shown for cifar100 but is the same for all other datasets):
```
datasets
└───cifar100
   └───train
       └───labelled
           └───apple
             │   <img1>.png
             │   <img2>.png
             │   ...
           └───aquarium_fish
             │   <img1>.png
             │   <img2>.png
             │   ...
           ...
           
       └───unlabelled
           └───UNKNOWN_CLASS
             │   <img1>.png
             │   <img2>.png
             │   <img3>.png
   └───val
       └───apple
             │   <img1>.png
             │   <img2>.png
             │   ...
       └───aquarium_fish
             │   <img1>.png
             │   <img1>.png
             │   ...
           ...
```

To preprocess a generic dataset into the above format, you can refer to `utils/data_utils.py` for several examples.

To configure the datasets directory path, you can either set the environment variable `SEMCO_DATA_PATH` or pass a command line argument `--dataset-path` to the launcher. (e.g. `export SEMCO_DATA_PATH=/home/data`). Note that this path references the parent datasets directory which contains the different sub directories for the individual datasets (e.g. cifar100, mini-imagenet, etc.)

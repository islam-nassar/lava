from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
METADATASET_REPO = os.environ.get('METADATASET_REPO', '/home/inas0003/projects/meta-dataset')
import sys
sys.path.append(METADATASET_REPO)

import os
from collections import Counter
import gin
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
import meta_dataset.learn.gin.setups

from pathlib import Path
from PIL import Image

METADATASET_ROOT = os.environ.get('METADATASET_ROOT', '/data/lwll/inas0003/meta_dataset/')
BASE_PATH = os.path.join(METADATASET_ROOT, 'RECORDS')
GIN_FILE_PATH = os.path.join(METADATASET_REPO, 'meta_dataset/learn/gin/setups/data_config.gin')
gin.parse_config_file(GIN_FILE_PATH)
ALL_DATASETS = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012',
                'omniglot', 'quickdraw', 'vgg_flower', 'traffic_sign', 'mscoco']

def get_dataset_specs(dataset_name_list):
    all_dataset_specs = []
    for dataset_name in dataset_name_list:
        dataset_records_path = os.path.join(BASE_PATH, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        all_dataset_specs.append(dataset_spec)
    return all_dataset_specs

def get_episode_iterator(dataset_name_list, split):
    SPLIT = learning_spec.Split[split.upper()]
    all_dataset_specs = get_dataset_specs(dataset_name_list)
    use_bilevel_ontology_list = [False] * len(dataset_name_list)
    use_dag_ontology_list = [False] * len(dataset_name_list)
    # Enable ontology aware sampling for Omniglot and ImageNet.
    if 'ilsvrc_2012' in dataset_name_list:
        use_dag_ontology_list[dataset_name_list.index('ilsvrc_2012')] = True
    if 'omniglot' in dataset_name_list:
        use_bilevel_ontology_list[dataset_name_list.index('omniglot')] = True

    variable_ways_shots = config.EpisodeDescriptionConfig(
        num_query=None, num_support=None, num_ways=None)

    if len(dataset_name_list) > 1:
        dataset_episodic = pipeline.make_multisource_episode_pipeline(
            dataset_spec_list=all_dataset_specs,
            use_dag_ontology_list=use_dag_ontology_list,
            use_bilevel_ontology_list=use_bilevel_ontology_list,
            episode_descr_config=variable_ways_shots,
            split=SPLIT,
            image_size=224,
            shuffle_buffer_size=300)

    elif len(dataset_name_list) == 1:
        dataset_episodic = pipeline.make_one_source_episode_pipeline(
            dataset_spec=all_dataset_specs[0], split=SPLIT,
            use_dag_ontology=use_dag_ontology_list[0],
            use_bilevel_ontology=use_bilevel_ontology_list[0],
            episode_descr_config=variable_ways_shots,
            image_size=224,
            shuffle_buffer_size=300)
    else:
        raise Exception('dataset_name_list len must be >= 1')

    return iter(dataset_episodic)


def write_episode_to_disk(episode, dataset_name, path, repetitions=0):
    all_dataset_specs = get_dataset_specs([dataset_name])
    class_names = np.array(list(all_dataset_specs[0].class_names.values()))
    pth_tr_lab = (Path(path) / 'train/labelled')
    pth_tr_lab.mkdir(parents=True, exist_ok=False)
    if repetitions:
        pth_tr_lab_no_rep = (Path(path) / 'train_no_rep/labelled')
        pth_tr_lab_no_rep.mkdir(parents=True, exist_ok=False)
    pth_tr_unlab = (Path(path) / 'train/unlabelled/UNKNOWN_CLASS')
    pth_tr_unlab.mkdir(parents=True, exist_ok=False)
    pth_val = (Path(path) / 'val')
    pth_val.mkdir(parents=True, exist_ok=False)
    pth_test = (Path(path) / 'test/UNKNOWN_CLASS')
    pth_test.mkdir(parents=True, exist_ok=False)

    episode = [elem.numpy() for elem in episode]
    # write support dataset under train labelled
    ctr = 0
    for img, cls in zip(episode[0], episode[2]):
        # scaling in line 92 in decoder.py was --> image = 2 * (image_resized / 255.0 - 0.5)  # Rescale to [-1, 1].
        # unscale image
        img = np.array(255 * (0.5 * img + 0.5), dtype='uint8')
        class_name = class_names[cls]
        if class_name not in os.listdir(pth_tr_lab):
            (pth_tr_lab / class_name).mkdir(parents=True, exist_ok=True)
            if repetitions:
                (pth_tr_lab_no_rep / class_name).mkdir(parents=True, exist_ok=True)

        # write image to disk
        img_path = (pth_tr_lab / class_name / f'{ctr:05}.jpg')
        Image.fromarray(img).save(str(img_path))
        if repetitions:
            os.symlink(img_path, (pth_tr_lab_no_rep / class_name / f'{ctr:05}.jpg'))
        # repeat if applicable
        for rep in range(repetitions - 1):
            os.symlink(img_path, str(img_path).replace('.jpg', f'_rep{rep:03}.jpg'))
        ctr += 1
    # write query dataset under val
    ctr = 0
    for img, cls in zip(episode[3], episode[5]):
        # scaling in line 92 in decoder.py was --> image = 2 * (image_resized / 255.0 - 0.5)  # Rescale to [-1, 1].
        # unscale image
        img = np.array(255 * (0.5 * img + 0.5), dtype='uint8')
        class_name = class_names[cls]
        if class_name not in os.listdir(pth_val):
            (pth_val / class_name).mkdir(parents=True, exist_ok=True)
        # write image to disk
        Image.fromarray(img).save(str(pth_val / class_name / f'{ctr:05}.jpg'))
        ctr += 1


def plot_episode(episode, size_multiplier=1, max_imgs_per_col=10,
                 max_imgs_per_row=10):

    episode = [a.numpy() for a in episode]
    support_images = episode[0]
    support_class_ids = episode[2]
    query_images = episode[3]
    query_class_ids = episode[5]

    for name, images, class_ids in zip(('Support', 'Query'),
                                       (support_images, query_images),
                                       (support_class_ids, query_class_ids)):
        n_samples_per_class = Counter(class_ids)
        n_samples_per_class = {k: min(v, max_imgs_per_col)
                               for k, v in n_samples_per_class.items()}
        id_plot_index_map = {k: i for i, k
                             in enumerate(n_samples_per_class.keys())}
        num_classes = min(max_imgs_per_row, len(n_samples_per_class.keys()))
        max_n_sample = max(n_samples_per_class.values())
        figwidth = max_n_sample
        figheight = num_classes
        if name == 'Support':
            print('#Classes: %d' % len(n_samples_per_class.keys()))
        figsize = (figheight * size_multiplier, figwidth * size_multiplier)
        fig, axarr = plt.subplots(
            figwidth, figheight, figsize=figsize)
        fig.suptitle('%s Set' % name, size='20')
        fig.tight_layout(pad=3, w_pad=0.1, h_pad=0.1)
        reverse_id_map = {v: k for k, v in id_plot_index_map.items()}
        for i, ax in enumerate(axarr.flat):
            ax.patch.set_alpha(0)
            # Print the class ids, this is needed since, we want to set the x axis
            # even there is no picture.
            ax.set(xlabel=reverse_id_map[i % figheight], xticks=[], yticks=[])
            ax.label_outer()
        for image, class_id in zip(images, class_ids):
            # First decrement by one to find last spot for the class id.
            n_samples_per_class[class_id] -= 1
            # If class column is filled or not represented: pass.
            if (n_samples_per_class[class_id] < 0 or
                    id_plot_index_map[class_id] >= max_imgs_per_row):
                continue
            # If width or height is 1, then axarr is a vector.
            if axarr.ndim == 1:
                ax = axarr[n_samples_per_class[class_id]
                if figheight == 1 else id_plot_index_map[class_id]]
            else:
                ax = axarr[n_samples_per_class[class_id], id_plot_index_map[class_id]]
            ax.imshow(image / 2 + 0.5)
        plt.show()

if __name__ == '__main__':
    print('Starting')

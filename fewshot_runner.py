# Author: Islam Nassar (islam.nassar@monash.edu)
# This file runs few-shot episodes based on the meta-dataset framework
import argparse
import os.path
import re
import shutil

from main_lava import get_args_parser as get_lava_parser
from main_lava import train_lava
from metadataset_utils import get_episode_iterator, write_episode_to_disk
import utils as utils

from pathlib import Path
import pandas as pd

def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def get_args_parser():
    parser = argparse.ArgumentParser('FSL_runner', add_help=False)

    # Model parameters
    parser.add_argument('--dataset_name', default=None, type=str,
        choices=['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012',
                'omniglot', 'quickdraw', 'vgg_flower', 'traffic_sign', 'mscoco'],
        help="""Meta Dataset name to run the fewshot experiment""")
    parser.add_argument('--num_episodes', default=10, type=int, help="""Number of episodes to run""")
    parser.add_argument('--repetitions', default=50, type=int, help="""Number of times to repeat the episode images. 
    This is to speed up the episode runtime and avoid very short epochs which lead to time overhead. The actual 
    num epochs and eval_freq will also be downscaled with this number""")
    parser.add_argument('--meta_model_weights',
                        default=None,
                        type=str, help="Path to pretrained weights of meta-trained model")
    parser.add_argument('--write_predictions_to_disk', default=False, type=utils.bool_flag,
                        help="""If set, each episode's prediction will be written to disk in the final epoch""")
    parser.add_argument('--meta_output_dir', default=None, type=str, help="""This is the root directory where the 
    fewshot experiment will run. Under this directory, there will be subdir with dataset names then subdir with episode ids""")
    parser.add_argument('--fewshot_config_path',
                        default='./aux/fewshot_config_imgnt_only.txt',
                        type=str, help="""This is the root directory where the 
        fewshot experiment will run. Under this directory, they will be subdir with dataset names then subdir with episode ids""")

    return parser


if __name__ == '__main__':
    parser_lava = get_lava_parser()
    args_lava, _ = parser_lava.parse_known_args()
    parser = argparse.ArgumentParser('FSL_runner', parents=[get_args_parser()])
    args, _ = parser.parse_known_args()

    fewshot_config = Path(args.fewshot_config_path).open('r').read().split('\n')
    fewshot_config = {elem.split(':')[0].strip() : elem.split(':')[1].strip() for elem in fewshot_config}
    for k,v in fewshot_config.items():
        if getattr(args_lava, k) is None:
            continue
        try:
            if v == 'False':
                v = False
            else:
                v = type(getattr(args_lava, k))(v)
        except TypeError:
            pass
        setattr(args_lava, k, v)

    args_lava.pretrained_weights = args.meta_model_weights
    args_lava.dist_url = 'env://'
    args_lava.local_crops_scale = (0.05, 0.4)
    args_lava.global_crops_scale = (0.4, 1.0)
    args_lava.write_predictions_to_disk = args.write_predictions_to_disk
    if args.repetitions:
        args_lava.epochs = max(1, args_lava.epochs // args.repetitions)
        args_lava.break_epoch = max(1, args_lava.break_epoch // args.repetitions)
        args_lava.eval_freq = max(1, args_lava.eval_freq // args.repetitions)
        args_lava.warmup_epochs = max(1, args_lava.warmup_epochs // args.repetitions)
        args_lava.warmup_teacher_temp_epochs = max(1, args_lava.warmup_teacher_temp_epochs // args.repetitions)
    episode_iterator = get_episode_iterator([args.dataset_name], 'test')
    (Path(args.meta_output_dir)/args.dataset_name).mkdir(parents=True, exist_ok=True)
    res_path = Path(args.meta_output_dir)/args.dataset_name/'results_df.csv'
    if os.path.exists(res_path):
        results_df = pd.read_csv(res_path)
    else:
        results_df = None
    for i in range(args.num_episodes):
        out_dir = (Path(args.meta_output_dir) / f'{args.dataset_name}/episode{i:03}')
        if os.path.exists(out_dir):
            print(f'{out_dir.name} already exists, skipping to next..')
            continue
        episode, s_id = episode_iterator.get_next()
        ep_path = Path(args.meta_output_dir)/f'{args.dataset_name}/episode_data'
        if os.path.exists(ep_path):
            shutil.rmtree(ep_path)
        # extract episode stats (N-way, K_c shot)
        num_samples_support = len(episode[0])
        num_samples_query = len(episode[3])
        num_way = len(set(episode[1].numpy()))
        # write to disk
        write_episode_to_disk(episode, dataset_name=args.dataset_name, path=ep_path, repetitions=args.repetitions)

        # set args for lava
        args_lava.data_path = str(Path(args.meta_output_dir) / f'{args.dataset_name}/episode_data')
        out_dir.mkdir(parents=True, exist_ok=False)
        args_lava.output_dir = str(out_dir)
        args_lava.dist_master_port = str(find_free_port())
        train_lava(args_lava)
        # get results from log.txt
        file = (out_dir/'log.txt').open('r').read()
        acc = re.findall(r'"val_acc1": (\d+\.\d+)', file)[-1]
        sem_acc = re.findall(r'"val_acc1_sem": (\d+\.\d+)', file)[-1]
        knn_acc = re.findall(r'"val_KNN_top1": (\d+\.\d+)', file)[-1]
        fusion = re.findall(r'"top1_fused": (\d+\.\d+)', file)[-1]
        res = pd.DataFrame({'episode': f'{i:03}', 'dataset': args.dataset_name, 'ways': num_way,
                            'num_samples_support': num_samples_support, 'num_samples_query': num_samples_query,
                            'one_hot_accuracy': acc, 'semantic_accuracy': sem_acc, 'knn_accuracy': knn_acc,
                            'fusion_accuracy': fusion}, index=[i])
        if results_df is None:
            results_df = res
        else:
            results_df = pd.read_csv(res_path)  # read again to make sure it is up-to-date in case of concurrent runs
            results_df = pd.concat([results_df, res], axis=0)

        results_df.to_csv(str(res_path), index=False)
        # clean up
        for f in out_dir.iterdir():
            if 'checkpoint' in f.name:
                os.remove(f)







